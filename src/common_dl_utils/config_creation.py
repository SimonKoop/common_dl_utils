""" 
Module providing convenient tools for the creation of (nested) config dicts for creating models etc.

The primary tools it provides are

Config: a convenience wrapper around a dictionary to allow attribute like access 
VariableCollector: used for organizing variables for grid-search, or for use in wandb sweeps
make_flat_config: because wandb sweeps are easiest to setup with flat configs (not nested)
make_nested_config: because the rest of this package assumes nested configs
    and nested configs are more flexible in terms of parameter name reuse and can be more readable in general

The two main use cases for this are the following:

1) without wandb
without the use of wandb, this class allows you to easily setup grid-searches
E.G. 
>>> variable = VariableCollector()
>>> config_template = Config()
>>> config_template.latent_size = variable(8, 16, 32, 64, group='latent_size')
>>> config_template.hidden_size_first_layer_decoder = variable(*(2*size for size in config_template.latent_size.values), group='latent_size')
>>> config_template.hidden_size_decoder = variable(128, 256)
>>> config_template.encoder_config = Config()
>>> config_template.encoder_config.hidden_size = variable(128, 256)
>>> config_template.encoder_config.hidden_size_final_layer = variable(*(2*size for size in config_template.latent_size.values), group=config_template.latent_size.group)  # also works without naming the group

one can get configs from this containing concrete realizations of these variables through variable.realizations(config_template)
In the above example this wil give you 4*2*2=16 configs (the variables in the same group are linked together)

2) with wandb
when using wandb sweeps, one can use VariableCollector when setting up grid search, or if a random sweep is required instead, one can give kwargs like so:
>>> variable = VariableCollector()
>>> config_template = Config()
>>> config_template.optimiser = 'Adam'
>>> config_template.optimiser_kwargs = {
    'lr': variable(distribution='log_uniform_values', min=1e-5, max=1e-2),
    'weight_decay': variable(1e-3, 5e-3, 1e-2, probabilities=[.5, .3, .2])
}
>>> config_template.encoder_use_batch_norm = True  # no need to explicitly put this as {'value': True}. This is handled by this module.
...
>>> sweep_parameters = make_flat_config(config_template)

NB when using wandb, the `group` key-word won't really do anything useful

In case 1) variable can contain sub-trees for the config that can themselves contain variables
in case 2) (wandb) that will not work with the wandb api


For smooth interoperability with wandb, use the make_flat_config and make_nested_config functions
"""
from collections import UserDict
from common_dl_utils import trees
import itertools
from typing import Union, List, Tuple, Dict, Any

__all__ = [
    "Config",
    "VariableCollector",
    "wandb_process_variable_token",
    "wandb_process_fixed_value",
    "make_flat_config",
    "make_nested_config",
]

class Config(UserDict):
    """
    Convenience class.
    Just a wrapper around UserDict to allow attribute like access besides key-like access.
    """
    _initialized = False

    def __init__(self, initial_data=None, /, **kwargs):
        super().__init__(initial_data, **kwargs)
        self._initialized = True

    def __getattr__(self, attr):
        if attr == 'data':
            # this might happen in loading with pickle
            raise AttributeError(attr)
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        if self._initialized:
            self[attr] = value
        else:
            object.__setattr__(self, attr, value)

    def __hasattr__(self, attr):
        return object.__hasattr__(self, attr) or attr in self

    def __delattr__(self, name):
        if not self.__hasattr__(name):
            raise AttributeError(f'Cannot delete attribute with name {name} because no such attribute exists')
        if name not in self.data:
            raise AttributeError(f'Cannot delete attribute with name {name} because it is not an item.')
        del self.data[name]

class _VariableToken:
    """ 
    Place holder for a variable

    Intended to be created by VariableCollector
    """
    def __init__(self, key_name, values, group, meta_data): 
        self.key_name = key_name 
        self.values = values
        self.group = group
        self.meta_data = meta_data

    def __call__(self, values: dict):
        return values[self.key_name]
    
    def __repr__(self):
        return f"_VariableToken(values={self.values}, group={self.group}, meta_data={repr(self.meta_data)})"
    
    def __str__(self):
        return self.__repr__()


class VariableCollector:
    """ 
    Helper class for making configs
    The two main use cases for this are the following:

    1) without wandb
    without the use of wandb, this class allows you to easily setup grid-searches
    E.G. 
    >>> variable = VariableCollector()
    >>> config_template = Config()
    >>> config_template.latent_size = variable(8, 16, 32, 64, group='latent_size')
    >>> config_template.hidden_size_first_layer_decoder = variable(*(2*size for size in config_template.latent_size.values), group='latent_size')
    >>> config_template.hidden_size_decoder = variable(128, 256)
    >>> config_template.encoder_config = Config()
    >>> config_template.encoder_config.hidden_size = variable(128, 256)
    >>> config_template.encoder_config.hidden_size_final_layer = variable(*(2*size for size in config_template.latent_size.values), group=config_template.latent_size.group)  # also works without naming the group

    one can get configs from this containing concrete realizations of these variables through variable.realizations(config_template)
    In the above example this wil give you 4*2*2=16 configs (the variables in the same group are linked together)

    2) with wandb
    when using wandb sweeps, one can use VariableCollector when setting up grid search, or if a random sweep is required instead, one can give kwargs like so:
    >>> variable = VariableCollector()
    >>> config_template = Config()
    >>> config_template.optimiser = 'Adam'
    >>> config_template.optimiser_kwargs = {
        'lr': variable(distribution='log_uniform_values', min=1e-5, max=1e-2),
        'weight_decay': variable(1e-3, 5e-3, 1e-2, probabilities=[.5, .3, .2])
    }
    >>> config_template.encoder_use_batch_norm = True  # no need to explicitly put this as {'value': True}. This is handled by this module.
    ...
    >>> sweep_parameters = make_flat_config(config_template)

    NB when using wandb, the `group` key-word won't really do anything useful
    """
    def __init__(self):
        self._group_to_id = {}
        self._group_to_lengths = {}
        self._id_to_group = {}
        self._id_to_values = {}

        self._id_generator = itertools.count()
        self._group_generator = itertools.count()

    def _get_next_group(self):
        while True:
            proposal = next(self._group_generator)
            if proposal not in self._group_to_id:
                return proposal 

    def __call__(self, *values, group=None, **kwargs)->_VariableToken:
        """ 
        Creates a new _VariableToken
        """
        if group is None:
            group = self._get_next_group()
        
        id = next(self._id_generator)
        
        if group in self._group_to_id:
            new_length = len(values)
            existing_length = self._group_to_lengths[group]
            if new_length != existing_length:
                raise ValueError(f"existing variables in group {group} take on {existing_length} different values, received {new_length} values for new variable.")
            self._group_to_id[group].append(id)
        else:
            self._group_to_id[group] = [id]
            self._group_to_lengths[group] = len(values)

        self._id_to_group[id] = group

        self._id_to_values[id] = list(values)

        return _VariableToken(id, values, group, kwargs)

    def __iter__(self):
        def _iterator():
            product = itertools.product(
                *(range(l) for l in self._group_to_lengths.values())
            )

            for indices in product:
                yield {
                    id: self._id_to_values[id][index]
                    for group, index in zip(self._group_to_id.keys(), indices)
                    for id in self._group_to_id[group]
                }
        return _iterator()

    def realizations(self, container):
        """
        :param container: some (possibly nested) container, containing among other things VariableToken instances
        :return: iterator of all possible realizations of said container. 
        """
        def _iterator():
            for key_value_dict in self:
                yield trees.repeated_tree_map(
                    tree=container,
                    func=lambda leaf: leaf(key_value_dict) if isinstance(leaf, _VariableToken) else leaf,
                    requires_repetition=lambda x: isinstance(x, _VariableToken)
                )
        return _iterator()
    

# the following two functions are so we can easily use the tools in this module
# for creating wandb sweeps.
def wandb_process_variable_token(token: _VariableToken):
    return_value = {}
    return_value.update(token.meta_data)
    if token.values:
        if len(token.values) > 1:
            return_value['values'] = token.values 
        else:
            return_value['value'] = token.values[0]
    return return_value

def wandb_process_fixed_value(value):
    return {'value': value}

# for working with wandb sweeps, we want the configs to be flat
# but nested configs can be useful for keeping things tidy and using the same parameter name (such as 'hidden_size') in multiple places without conflicts
# so the following functions are here to bridge the gap between flat and nested configs 
def make_flat_config(container, path_sep='__', variables_only=False):
    """ 
    make a flat config dict compatible with wandb out of the (meta data of) the collected variables

    :param container: nested container (ideally dicts or user dicts) with among other things _VariableToken instances
    :param path_sep: character used to separate indices of nested dicts
    :param variables_only: if True, we generate a config containing only the variables
        if False, we add all the fixed values to the wandb config too

    NB will not work for _VariableTokens containing sub-configs 
    """
    if variables_only:
        paths_and_variables = trees.get_items_of_interest(container, is_value_of_interest=lambda x: isinstance(x, _VariableToken))

        return {
            path_sep.join((str(index) for index in path)): wandb_process_variable_token(variable)
            for path, variable in paths_and_variables
        }
    else:
        paths_and_everything = trees.get_items_of_interest(container, is_value_of_interest=lambda x: not isinstance(x, (dict, UserDict, list, tuple)))
        return_value = {}
        for path, value in paths_and_everything:
            path = path_sep.join((str(index) for index in path))
            if isinstance(value, _VariableToken):
                return_value[path] = wandb_process_variable_token(value)
            else: 
                return_value[path] = wandb_process_fixed_value(value)
        return return_value
    
def _add_to_container(key: str, container: Union[List, Config, Dict], value: Any):
    if isinstance(container, (Config, dict)):
        container[key] = value 
    else:
        key = int(key)
        if len(container) != key:
            raise ValueError(f"Can't add to list in arbitrary order: {container}, {key=}") 
        container.append(value)

def _merge_key_path_into_container(key_path: Union[List[str], Tuple[str]], container: Union[list, Config, Dict], value: Any):
    """ 
    Try to merge a key_path into a container.
    NB it is assumed that any integer keys belong to lists
    """
    if len(key_path) == 1:
        _add_to_container(key=key_path[0], container=container, value=value)
    elif isinstance(container, (Config, dict)):
        key = key_path[0]
        if key not in container:
            if key_path[1].isdecimal():
                container[key] = list()
            else:
                container[key] = Config()
        _merge_key_path_into_container(key_path[1:], container[key], value)
    else:
        key = int(key_path[0])
        if key == len(container):
            if key_path[1].isdecimal():
                container.append(list())
            else:
                container.append(Config())

        if key < len(container):  #NB now includes previous case
            _merge_key_path_into_container(key_path[1:], container[key], value)
        else:
            raise ValueError(f"Can't add to list in arbitrary order: {container}, {key=}")

def make_nested_config(flat_config, path_sep='__'):
    """ 
    make a nested config out of the flat config from wandb
    """
    config = Config() 

    for key_path_str, value in flat_config.items():
        key_path = key_path_str.split(path_sep)
        _merge_key_path_into_container(key_path, config, value)
    return config 
