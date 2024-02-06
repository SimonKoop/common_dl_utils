""" 
Module for automatically creating models from config dicts.
Many of the models in the survey contain multiple sub-models. 
Initializing a full model requires initialization of all its sub models.
Hardcoding this for every model makes swapping out architectures and training loops etc. cumbersome 
Instead, it is useful to give the full specification of the model with all its submodels and the training loop in terms of only 
- what classes/functions to use (and where to find them)
- what parameters to initialize them with

This module allows you to do that.

Example (placed outside of the util directory): 
>>> config_1 = {
    "model_type": 'TestType2',  
    "architecture": './testing/test_classes.py',
    "TestType2_config": {
        'param_1': 121,
        'param_2': 122,
        'toggle_1': True, 
        'sub_model_tt2': 'TestType1'
    },
    "TestType1_config":{
        'param_1': 111,
        'param_2': 112,
        'toggle_1': True,
        'toggle_2': False
    }
}
>>> print(utils.config_realization.get_model_from_config(
        config=config_1,
    ))

# output: TestType2(param_1=121, param_2=122, toggle_1=True, sub_model_tt2=TestType1(param_1=111, param_2=112, toggle_1=True, toggle_2=False))

or in combination with config_creation.py

>>> from common_dl_utils.config_creation import VariableCollector, Config
>>> variable = VariableCollector()
>>> config = Config()
>>> config.model_type = 'TestType2'
>>> config.architecture = './testing/test_classes.py'
>>> config.TestType2_config = Config(
>>>     param_1 = variable(121, 112211, group='p1'), 
>>>     param_2 = 122,
>>>     toggle_1 = True,
>>>     sub_model_tt2 = variable('TestType1', 'TestType3', group='submodel')
>>> )
>>> config.sub_model_tt2_config = variable(
>>>     Config(
>>>         param_1 = variable(111, 111111, group='p1'),
>>>         param_2 = 112,
>>>         toggle_1 = variable(True, False),
>>>         toggle_2 = False
>>>     ),
>>>     Config(
>>>         param_1 = variable(131, 113311, group='p1'),
>>>         param_2 = 132,
>>>         toggle_1=variable(True, False),
>>>         toggle_2 = False,
>>>         sub_model_tt3='TestType1',
>>>         sub_model_tt3_config=Config(
>>>             param_1 = variable(111, 111111, group='p1'),
>>>             param_2 = 112,
>>>             toggle_1 = True,
>>>             toggle_2 = False
>>>         )
>>>     ),
>>>     group='submodel'
>>> )
>>> 
>>> 
>>> for c in variable.realizations(config):
>>>     #pprint(c)
>>>     print(utils.config_realization.get_model_from_config(
>>>         config=c,
>>>         sub_config_from_param_name=True
>>>     ))

# output: 
TestType2(param_1=121, param_2=122, toggle_1=True, sub_model_tt2=TestType1(param_1=111, param_2=112, toggle_1=True, toggle_2=False))
TestType2(param_1=121, param_2=122, toggle_1=True, sub_model_tt2=TestType1(param_1=111, param_2=112, toggle_1=True, toggle_2=False))
TestType2(param_1=121, param_2=122, toggle_1=True, sub_model_tt2=TestType1(param_1=111, param_2=112, toggle_1=False, toggle_2=False))
TestType2(param_1=121, param_2=122, toggle_1=True, sub_model_tt2=TestType1(param_1=111, param_2=112, toggle_1=False, toggle_2=False))
TestType2(param_1=121, param_2=122, toggle_1=True, sub_model_tt2=TestType3(param_1=131, param_2=132, toggle_1=True, toggle_2=False, sub_model_tt3=TestType1(param_1=111, param_2=112, toggle_1=True, toggle_2=False)))
TestType2(param_1=121, param_2=122, toggle_1=True, sub_model_tt2=TestType3(param_1=131, param_2=132, toggle_1=False, toggle_2=False, sub_model_tt3=TestType1(param_1=111, param_2=112, toggle_1=True, toggle_2=False)))
TestType2(param_1=121, param_2=122, toggle_1=True, sub_model_tt2=TestType3(param_1=131, param_2=132, toggle_1=True, toggle_2=False, sub_model_tt3=TestType1(param_1=111, param_2=112, toggle_1=True, toggle_2=False)))
TestType2(param_1=121, param_2=122, toggle_1=True, sub_model_tt2=TestType3(param_1=131, param_2=132, toggle_1=False, toggle_2=False, sub_model_tt3=TestType1(param_1=111, param_2=112, toggle_1=True, toggle_2=False)))
TestType2(param_1=112211, param_2=122, toggle_1=True, sub_model_tt2=TestType1(param_1=111111, param_2=112, toggle_1=True, toggle_2=False))
TestType2(param_1=112211, param_2=122, toggle_1=True, sub_model_tt2=TestType1(param_1=111111, param_2=112, toggle_1=True, toggle_2=False))
TestType2(param_1=112211, param_2=122, toggle_1=True, sub_model_tt2=TestType1(param_1=111111, param_2=112, toggle_1=False, toggle_2=False))
TestType2(param_1=112211, param_2=122, toggle_1=True, sub_model_tt2=TestType1(param_1=111111, param_2=112, toggle_1=False, toggle_2=False))
TestType2(param_1=112211, param_2=122, toggle_1=True, sub_model_tt2=TestType3(param_1=113311, param_2=132, toggle_1=True, toggle_2=False, sub_model_tt3=TestType1(param_1=111111, param_2=112, toggle_1=True, toggle_2=False)))
TestType2(param_1=112211, param_2=122, toggle_1=True, sub_model_tt2=TestType3(param_1=113311, param_2=132, toggle_1=False, toggle_2=False, sub_model_tt3=TestType1(param_1=111111, param_2=112, toggle_1=True, toggle_2=False)))
TestType2(param_1=112211, param_2=122, toggle_1=True, sub_model_tt2=TestType3(param_1=113311, param_2=132, toggle_1=True, toggle_2=False, sub_model_tt3=TestType1(param_1=111111, param_2=112, toggle_1=True, toggle_2=False)))
TestType2(param_1=112211, param_2=122, toggle_1=True, sub_model_tt2=TestType3(param_1=113311, param_2=132, toggle_1=False, toggle_2=False, sub_model_tt3=TestType1(param_1=111111, param_2=112, toggle_1=True, toggle_2=False)))

For getting classes from other files, just put a prompt of the form
>>> (path, class_name) 
to the config dict, where path is either a relative, or an absolute path (e.g. ("./architectures/dataset_1/v1.py", 'Encoder'))

High level function: get_model_from_config
Low level function that requires a little bit more boiler plate but is more broadly applicable: prep_class_from_config

NB wherever it says class, typically any type of callable should work
"""

import inspect
from typing import Union, Callable, Any, get_origin, get_args
#from collections.abc import Sequence
from types import ModuleType
from common_dl_utils.type_registry import type_registry, contains, is_in_registry
from common_dl_utils.module_loading import load_from_path, MultiModule
from common_dl_utils.trees import get_from_index, has_index, tree_map
import common_dl_utils._internal_utils as _internal_utils
from collections.abc import Iterable
from functools import partial
import os
import warnings

__all__ = [
    'Prompt',
    'ExtendedPrompt',
    'PostponedInitialization',
    'generalized_getattr',
    'process_prompt',
    'split_extended_prompt',
    'prep_class_from_extended_prompt',
    'prep_class_from_config',
    'get_parameter_from_config',
    'get_callable_from_extended_prompt',
    'match_signature_from_config',
    'get_model_from_config'
]

# common types
Prompt = Union[tuple[str, str], list[str, str], str, Callable, None]
ExtendedPrompt = Union[tuple[str, str, dict], list[str, str, dict], tuple[str, dict], list[str, dict], tuple[Callable, dict], list[Callable, dict], Prompt]
Prompt.__doc__ = """
Prompt (type alias)
either a string specifying the name of some object in some default_module
or a tuple/list of two strings, 
    the first of which specifying the location of the appropriate module, 
    and the second being the name of the required object within that module
or a callable to be used directly
or None if this is for an optional argument
"""
ExtendedPrompt.__doc__ = """
ExtendedPrompt (type alias)
Either a Prompt (see Prompt) or one of the following:
    a tuple/list with one or two strings and a dict
        in case of one string it is the name of an object in some default_module
        in case of two strings, the first is the location of the appropriate module, the second the name of an object in said module
        the dict is a local_config for the object (specifying parameters for initialization/calling)
    a tuple/list with a Callable and a dict
        the dict is again a local_config for the Callable
"""


class PostponedInitialization:
    """ 
    to prevent unused models from being created in while looping over the keys in match_signature_from_config,
    we postpone initialization of classes based on kwargs until we are certain we need them
    """ # TODO handle VAR_POSITIONAL
    def __init__(self, cls:type, kwargs:dict, missing_args:Union[None, list]=None, signature:inspect.Signature=None, prompt: Prompt=None, associated_parameter_name:Union[None, str]=None): 
        self.cls = cls 
        self.signature = signature if signature is not None else inspect.signature(cls)
        self.kwargs = kwargs 
        self.missing_args=missing_args
        self.needs_wrapping = any(param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.POSITIONAL_ONLY) for param in self.signature.parameters.values())
        self.prompt=prompt  # so we know from what prompt this was
        self.associated_parameter_name = associated_parameter_name
    
    def initialize(self, **kwargs):
        if self.missing_args and any(missing_arg not in kwargs for missing_arg in self.missing_args):
            raise KeyError(f"Arguments missing for {self.cls.__name__}: {[missing_arg for missing_arg in self.missing_args if missing_arg not in kwargs]}")
        # processed_self_kwargs = {
        #     key: value.initialize() if isinstance(value, PostponedInitialization) else value
        #     for key, value in self.kwargs.items()
        # }
        # the above was when we didn't support varargs, 
        # now that we do, we might have tuples or lists containing PostponedInitialization instances
        processed_self_kwargs = tree_map(
            tree = self.kwargs,
            func = lambda x: x.initialize() if isinstance(x, PostponedInitialization) else x,
            is_leaf = lambda x: isinstance(x, PostponedInitialization)  # this line is really optional
        )
        # allow kwargs to contain PostponedInitialization instances too
        processed_kwargs = tree_map(
            tree = kwargs,
            func = lambda x: x.initialize() if isinstance(x, PostponedInitialization) else x,
            is_leaf = lambda x: isinstance(x, PostponedInitialization) 
        )
        processed_self_kwargs.update(processed_kwargs)
        cls = _internal_utils.signature_to_var_keyword(self.cls, self.signature) if self.needs_wrapping else self.cls
        return cls(**processed_self_kwargs)
    
    def __repr__(self) -> str:
        return f"PostponedInitialization(cls={self.cls.__name__}, kwargs={self.kwargs}, missing_args={self.missing_args})"
    def __str__(self) -> str:
        return self.__repr__()
    
    def resolve_missing_args(self, resolution:Union[dict, 'PostponedInitialization'], allow_different_class:bool=False):
        if isinstance(resolution, PostponedInitialization):
            if self.cls is not resolution.cls and not allow_different_class:
                raise ValueError(f"{self.cls=} does not match {resolution.cls=} and {allow_different_class=}")
            resolution = resolution.kwargs
        if not self.missing_args:
            return None
        for arg_name in resolution:  # resolution is now a dict
            if arg_name in self.missing_args:
                self.missing_args.remove(arg_name)
                self.kwargs[arg_name] = resolution[arg_name]

    def is_complete(self)-> bool:
        """
        is_complete recursively checks whether all arguments have been resolved
        """
        no_missing_args = not self.missing_args
        all_simple_children_complete = all(child.is_complete() for child in self.kwargs.values() if isinstance(child, PostponedInitialization))
        all_list_or_tuple_children_complete = all(
            all(
                element.is_complete() 
                for element in child if isinstance(element, PostponedInitialization)
            )
            for child in self.kwargs.values() 
            if isinstance(child, (list, tuple))
        )
        return no_missing_args and all_simple_children_complete and all_list_or_tuple_children_complete

def maybe_add_key(key, keys, config):
    """ 
    add key to keys if it isn't already in there and it is a valid index for config
    """
    if has_index(config, key) and key not in keys:
        keys.append(key)

def generalized_getattr(
        obj:Any, 
        attr:Union[str, tuple[str], list[str]],
        default:Any=None,
        strict: bool=False
        ):
    """ 
    potentially repeatedly apply getattr where attr may be a dotted path
    :param obj: object to get attribute from
    :param attr: (path to) attribute to get from obj
    :param default: default value to return if attribute is not found
    :param strict: if True, raise an AttributeError if any earlier part of the path refers to a non existing attribute
        e.g. attempting to get a.b.c as generalized_getattr(a, 'b.c', strict=True) will raise an AttributeError if a.b does not exist
        On the other hand attempting to get a.b.c as generalized_getattr(a, 'b.c', strict=True) will return default if a.b.c does not exist but a.b. does, even if strict is True
    :raises AttributeError: if strict is True and an intermediate attribute is not found
    :return: the attribute if it exists, otherwise default
    """
    dotted_path = attr.split('.') if isinstance(attr, str) else attr
    if len(dotted_path) == 1:
        return getattr(obj, dotted_path[0], default)
    else:
        if not hasattr(obj, dotted_path[0]):
            if strict:
                raise AttributeError(f"{obj=} does not have attribute {dotted_path[0]}")
            else:
                return default
        return generalized_getattr(
            getattr(obj, dotted_path[0]), 
            dotted_path[1:],
            default,
            strict
            )

def process_prompt(
        prompt: Prompt,  # Union[tuple/list[str, str], str, Callable, None],
        default_module:Union[None, ModuleType, str]=None, 
        ):
    """process_prompt process a prompt from a config
    NB In the documentation and in the naming, prompt is assumed to point to a class
    however, in principle, it can point to any callable, including functions

    :param prompt: 
        either a string specifying the class name in the default_module
        or a tuple/list of strings, 
            the first of which specifying the location of the appropriate module, 
            and the second being the name of the class within that module
        or a callable to be used directly
        or None if this is for an optional argument
    :param default_module: optionally the default module from which to get classes if the prompt only specifies a class name
        either an actual module, 
        or a path to a module
    :raises ValueError: if no default module is provided and the prompt is only a string, a ValueError is raised
    :return: cls, class_name

    here cls refers to the callable the prompt points to
    and class_name refers to the best guess for the name of that callable
    if a dotted path is used, the class_name will be the first part of that dotted path
    """
    # we do an undocumented convenience thing here
    # because one might have a list like [(str, dict), (str, str, dict), str] in the config
    # and might be inclined to write it as [(str, dict), (str, str, dict), (str,)] out of stylistic consistency
    # and we don't want this to cause a needless Exception
    # but we also don't want to explicitly support tuple[str] type prompts
    # because it will make finding out whether we are dealing with a prompt, or a tuple of prompts more difficult
    if isinstance(prompt, (tuple, list)) and len(prompt)==1 and isinstance(prompt[0], str):
        prompt = prompt[0]

    if prompt is None:  # TODO write test case for this
        cls = None
        class_name = 'None'
    elif isinstance(prompt, str):
        if default_module is None:
            raise ValueError(f"missing path or class name in {prompt=} in absence of a default module")
        class_name = prompt
        module = default_module if isinstance(default_module, ModuleType) else load_from_path('default_module', default_module)
        cls = generalized_getattr(module, class_name)
    elif callable(prompt):  
        # mostly useful for when you're using this to create models
        # in notebooks directly, and you want to just feed classes and functions
        # to all the mechanics
        # TODO see if more documentation needs to be updated with this
        # this now also happens when a class requires a callable as an argument (e.g. activation function choice)
        cls = prompt
        class_name = cls.__name__
    else:
        path, class_name = prompt 
        module = load_from_path('_temp_module', path)
        cls = generalized_getattr(module, class_name)

    # in case of dotted path, make the class_name the first part of said path
    # e.g. 'Model.from_spec' becomes 'Model'
    class_name = class_name.split('.')[0]
    return cls, class_name

def split_extended_prompt(extended_prompt:ExtendedPrompt)->tuple[Prompt, dict]:
    """split_extended_prompt split an ExtendedPrompt into a Prompt and a dictionary (for local config)

    :param extended_prompt: one of:
        path, name, local_config
        name, local_config
        callable, local_config
        path, name
        name
        callable
        None
    :return: Prompt, local_config dict
    NB if no local_config is provided, an empty dict is returned
    """
    if isinstance(extended_prompt, Iterable) and isinstance(extended_prompt[-1], dict):
        prompt, local_config = extended_prompt[:-1], extended_prompt[-1]
        if len(prompt) == 1:
            prompt = prompt[0]
    else:
        prompt, local_config = extended_prompt, {}
    return prompt, local_config

def update_postponed_initialization_from_config(
        postponed_init:PostponedInitialization,
        config:dict,
        default_module:Union[None, ModuleType, str]=None,
        registry:list=type_registry,
        keys:Union[None, list]=None,
        current_key:Union[None, str, tuple[str]]=None,
        new_key_postfix:str='_config',
        new_key_body:Union[None, str]=None,
        new_key_base_from_param_name:bool = False,
        ignore_params:Union[None, list[str], tuple[str]] = None,
        ):
    """
    recursively update a PostponedInitialization instance if it or any of its children are missing arguments
    here, with children we mean any PostponedInitialization instances in the kwargs of the PostponedInitialization instance

    :param postponed_init: PostponedInitialization instance to update
    :param config: the config to get the missing arguments from
    :param default_module: optionally the default module from which to get classes if the prompt only specifies a class name
        either an actual module, 
        or a path to a module
    :param registry: list of types that require initialization (or just calling) based on config before being returned
    :param keys: (optional) potential children of config where values
        may be retrieved from. 
        Righter most is processed last.
        If B is processed after A, B can overrule the values in A. 
        A key can either be a string or a tuple of strings
        in case of a tuple of strings, it's treated as config[key[0]][key[1]]...
        a copy of keys is used instead and updated with f'{class_name}{new_key_postfix}' and potentially with f'{class_name}{new_key_postfix}' appended to current_key (and subkeys)
    :param current_key: (optional) update keys based on this to deal with nested configs
    :param new_key_postfix: string used for finding names of subconfigs  
    :param new_key_body: string to be used instead of class_name for the creation of new keys. If None, class_name is used
    :param new_key_base_from_param_name: if True, uses the parameter name to look for sub-configs instead of the class name
    :param ignore_params: optional collection of parameter names to be ignored in the process (will be added to missing_args)
    :return: None  as this is an in place operation
    """
    if postponed_init.is_complete():
        return None  # no updates needed
    
    # first get any missing arguments from config
    resolved_argument_names = list(postponed_init.kwargs.keys())
    if postponed_init.missing_args:
        postponed_init.resolve_missing_args(
            resolution=prep_class_from_config(
                prompt=postponed_init.prompt,
                config=config,
                default_module=default_module,
                registry=registry,
                keys=keys,
                current_key=current_key,
                new_key_postfix=new_key_postfix,
                new_key_body=new_key_body,
                new_key_base_from_param_name=new_key_base_from_param_name,
                ignore_params=ignore_params+resolved_argument_names
            )
        )

    # now recursively update any PostponedInitialization instances in the kwargs if needed
    for value in postponed_init.kwargs.values():
        if isinstance(value, PostponedInitialization):
            # we need to get the right new_key_body possibly based on the associated parameter name
            nkb = value.associated_parameter_name if new_key_base_from_param_name else None
            update_postponed_initialization_from_config(
                postponed_init=value,
                config=config,
                default_module=default_module,
                registry=registry,
                keys=_get_updated_keys(  # we need to do the update here to have it stick in the recursive calls
                    keys=keys,
                    current_key=current_key,
                    class_name=value.cls.__name__,
                    new_key_body=nkb,
                    new_key_postfix=new_key_postfix,
                    config=config
                ),
                current_key=current_key,
                new_key_postfix=new_key_postfix,
                new_key_body=nkb,
                new_key_base_from_param_name=new_key_base_from_param_name,
                ignore_params=ignore_params
            )
        # also update any lists and tuples of PostponedInitialization instances
        elif isinstance(value, (list, tuple)):
            for element in value:
                if isinstance(element, PostponedInitialization):
                    nkb = element.associated_parameter_name if new_key_base_from_param_name else None
                    update_postponed_initialization_from_config(
                        postponed_init=element,
                        config=config,
                        default_module=default_module,
                        registry=registry,
                        keys=_get_updated_keys(
                            keys=keys,
                            current_key=current_key,
                            class_name=element.cls.__name__,
                            new_key_body=nkb,
                            new_key_postfix=new_key_postfix,
                            config=config
                        ),
                        current_key=current_key,
                        new_key_postfix=new_key_postfix,
                        new_key_body=nkb,
                        new_key_base_from_param_name=new_key_base_from_param_name,
                        ignore_params=ignore_params
                    )


def prep_class_from_extended_prompt(
        extended_prompt: ExtendedPrompt,
        config: dict, 
        default_module:Union[None, ModuleType, str]=None, 
        registry:list=type_registry,
        keys:Union[None, list]=None,
        current_key:Union[None, str, tuple[str]]=None,
        new_key_postfix:str='_config',
        new_key_body:Union[None, str]=None,
        new_key_base_from_param_name:bool = False,
        ignore_params:Union[None, list[str], tuple[str]] = None, # TODO: test ignore_params
        param_name:Union[None, str]=None
        )->PostponedInitialization:
    """ 
    Prepare a class from an extended prompt

    :param extended_prompt: one of:
        path, name, local_config  
            where name should refer to a callable in the module located at path
        name, local_config  
            where name should refer to a callable in default_module
        callable, local_config
        path, name
        name
        callable
        None
    :param config: config dict containing parameters for initialization
        if the extended_prompt contains a local_config, we first try to prepare the class from that local_config
        if any arguments are then missing, we try to resolve these from config
    :param default_module: optionally the default module from which to get classes if the prompt only specifies a class name
        either an actual module, 
        or a path to a module
    :param registry: list of types that require initialization (or just calling) based on config before being returned
    :param keys: (optional) potential children of config where values
        may be retrieved from. 
        Righter most is processed last.
        If B is processed after A, B can overrule the values in A. 
        A key can either be a string or a tuple of strings
        in case of a tuple of strings, it's treated as config[key[0]][key[1]]...
        a copy of keys is used instead and updated with f'{class_name}{new_key_postfix}' and potentially with f'{class_name}{new_key_postfix}' appended to current_key (and subkeys)
    :param current_key: (optional) update keys based on this to deal with nested configs
    :param new_key_postfix: string used for finding names of subconfigs  
    :param new_key_body: string to be used instead of class_name for the creation of new keys. If None, class_name is used
    :param new_key_base_from_param_name: if True, uses the parameter name to look for sub-configs instead of the class name
    :param ignore_params: optional collection of parameter names to be ignored in the process (will be added to missing_args)
    :param param_name: (optional) name of the parameter that the class is being prepared for

    :returns: PostponedInitialization object

    NB In the documentation and in the naming, prompt is assumed to point to a class
    however, in principle, it can point to any callable, including functions
    """
    prompt, local_config = split_extended_prompt(extended_prompt)
    if ignore_params is None:
        ignore_params = []
    elif isinstance(ignore_params, tuple):
        ignore_params = list(ignore_params)

    if not local_config:  # just resort to prep_class_from_config
        return prep_class_from_config(
        prompt=prompt,
        config=config,
        default_module=default_module,
        registry=registry,
        keys=keys,
        current_key=current_key,
        new_key_postfix=new_key_postfix,
        new_key_body=new_key_body,
        new_key_base_from_param_name=new_key_base_from_param_name,
        ignore_params=ignore_params,
        param_name=param_name
    )

    # first try to get the class from the prompt and the local config
    postponed_init = prep_class_from_config(
        prompt=prompt,
        config=local_config,
        default_module=default_module,
        registry=registry,
        keys=None,
        current_key=None,
        new_key_postfix=new_key_postfix,
        new_key_body=new_key_body,
        new_key_base_from_param_name=new_key_base_from_param_name,
        ignore_params=ignore_params,
        param_name=param_name
    )
    # then we update the postponed_initialization with any missing arguments from the config
    update_postponed_initialization_from_config(
        postponed_init=postponed_init,
        config=config,
        default_module=default_module,
        registry=registry,
        keys=keys,
        current_key=current_key,
        new_key_postfix=new_key_postfix,
        new_key_body=new_key_body,
        new_key_base_from_param_name=new_key_base_from_param_name,
        ignore_params=ignore_params
    )
    return postponed_init

def _get_updated_keys(
        keys: Union[None, list],
        current_key:Union[None, str, tuple[str]],
        class_name:str,
        new_key_body:Union[None, str],
        new_key_postfix:str,
        config:dict
        ):
    """ 
    function for bookkeeping where to look for parameters in the config
    """
    new_key_body = class_name if new_key_body is None else new_key_body

    # update where to look for parameters in the config
    keys = list(keys) if keys is not None else []

    new_key_base = f'{new_key_body}{new_key_postfix}'
    maybe_add_key(new_key_base, keys, config)

    # if we're working in the context of a current_key, also add a multi-index based on it to the keys
    if isinstance(current_key, str):
        new_key = (current_key, new_key_base)
        maybe_add_key(new_key, keys, config)
    elif isinstance(current_key, tuple):
        for upto in range(1, len(current_key)+1):
            new_key = current_key[:upto] + (new_key_base, )
            maybe_add_key(new_key, keys, config)
    return keys

def prep_class_from_config(
        prompt: Prompt, #  Union[tuple/list[str, str], str, Callable, None], 
        config: dict, 
        default_module:Union[None, ModuleType, str]=None, 
        registry:list=type_registry,
        keys:Union[None, list]=None,
        current_key:Union[None, str, tuple[str]]=None,
        new_key_postfix:str='_config',
        new_key_body:Union[None, str]=None,
        new_key_base_from_param_name:bool = False,
        ignore_params:Union[None, tuple[str], list[str]] = None, # TODO: test ignore_params
        param_name:Union[None, str]=None,
        )->PostponedInitialization:
    """ 
    :param prompt: 
        either a string specifying the class name in the default_module
        or a tuple of strings, 
            the first of which specifying the location of the appropriate module, 
            and the second being the name of the class withing that module
        or a callable to be used directly
        or None if this is for an optional argument
    :param config: config dict containing parameters for initialization
    :param default_module: optionally the default module from which to get classes if the prompt only specifies a class name
        either an actual module, 
        or a path to a module
    :param registry: list of types that require initialization (or just calling) based on config before being returned
    :param keys: (optional) potential children of config where values
        may be retrieved from. 
        Righter most is processed last.
        If B is processed after A, B can overrule the values in A. 
        A key can either be a string or a tuple of strings
        in case of a tuple of strings, it's treated as config[key[0]][key[1]]...
        a copy of keys is used instead and updated with f'{class_name}{new_key_postfix}' and potentially with f'{class_name}{new_key_postfix}' appended to current_key (and subkeys)
    :param current_key: (optional) update keys based on this to deal with nested configs
    :param new_key_postfix: string used for finding names of subconfigs  
    :param new_key_body: string to be used instead of class_name for the creation of new keys. If None, class_name is used
    :param new_key_base_from_param_name: if True, uses the parameter name to look for sub-configs instead of the class name
    :param ignore_params: optional collection of parameter names to be ignored in the process (will be added to missing_args)
    :param param_name: (optional) name of the parameter that the class is being prepared for

    :returns: PostponedInitialization object

    NB In the documentation and in the naming, prompt is assumed to point to a class
    however, in principle, it can point to any callable, including functions
    """
    # first get the appropriate class from the appropriate module
    cls, class_name = process_prompt(prompt, default_module=default_module)
    # print(f"prepare {cls=} with {class_name=} from {prompt=}")
    if cls is None:
        return None
    
    # get the call signature for cls
    signature = inspect.signature(cls)

    # update where to look for parameters in the config
    keys = _get_updated_keys(
        keys=keys,
        current_key=current_key,
        class_name=class_name,
        new_key_body=new_key_body,
        new_key_postfix=new_key_postfix,
        config=config
    )
    
    # get the relevant values from the config
    kwargs, missing_args = match_signature_from_config(
        signature=signature, 
        config=config, 
        keys=keys, 
        registry=registry, 
        default_module=default_module, 
        new_key_postfix=new_key_postfix,
        new_key_base_from_param_name=new_key_base_from_param_name,
        ignore_params=ignore_params
        )

    return PostponedInitialization(cls, kwargs, missing_args, signature, prompt=prompt, associated_parameter_name=param_name)
    
    
def get_parameter_from_config(
        param_name:str, 
        details:inspect.Parameter, 
        config:dict, 
        registry:list=type_registry, 
        default_module:Union[None, ModuleType, str]=None, 
        keys:Union[None, list]=None, 
        sub_config:Union[None, dict]=None, 
        current_key:Union[None, str, tuple[str]]=None,
        new_key_postfix: str='_config',
        new_key_base_from_param_name:bool = False,
        ):
    """ 
    Get config[param_name]
    if config[param_name] needs to be initialized according to registry, prepare for doing so.
    if config[param_name] is supposed to be a callable or type according to the annotation, get the callable from the config
        if config[param_name] is an extended prompt containing a local_config, the callable is wrapped in a partial with that local_config
        NB if this extended prompt contains classes requiring initialization, the partial may contain PostponedInitialization instances in its stored arguments
            as it is unclear what the correct behaviour here should be (initialize once upon creation of the partial, or every time upon calling the partial)
            we leave this to the user to decide
    :param param_name: key for the config
    :param details: inspect.Parameter containing specifications about what param_name should contain (for checking against registry)
    :param config: config from which to get the value
    :param registry: list of types that require initialization (or just calling) based on config before being returned
    :param default_module: optionally the default module from which to get classes if the prompt only specifies a class name
        either an actual module, 
        or a path to a module
    :param keys: (optional) potential children of config where values
        may be retrieved from. 
        Righter most is processed last.
        If B is processed after A, B can overrule the values in A. 
        A key can either be a string or a tuple of strings
        in case of a tuple of strings, it's treated as config[key[0]][key[1]]...

        a copy of keys is used instead and updated with f'{class_name}_config' and potentially with f'{class_name}_config' appended to current_key
    :param sub_config: optional subconfig from which to get param_name instead. config will still be used for prep_class_from_config
    :param current_key: (optional) update keys based on this to deal with nested configs
    :param new_key_postfix: string used for finding names of subconfigs 
    :param new_key_base_from_param_name: if True, uses the parameter name to look for sub-configs instead of the class name
    """
    # first some checks on the type of the parameter
    # **kwargs are not supported because we don't really have a way of knowing what might be in there
    if details.kind is details.VAR_KEYWORD:
        raise NotImplementedError(f"We don't support matching VAR_KEYWORD arguments (raised in treating {param_name=} with {details=})")
    
    contents = config[param_name] if sub_config is None else sub_config[param_name]

    # there are a few different cases we need to handle
    # we'll first define how to handle them
    # then we'll define how to check for them
    # and then we'll execute the appropriate handler based on the check

    handle_registry_element_case_single = partial(
        prep_class_from_extended_prompt, 
        config=config, 
        default_module=default_module,
        registry=registry,
        keys=keys,
        current_key=current_key,
        new_key_postfix=new_key_postfix,
        new_key_body= param_name if new_key_base_from_param_name else None,
        new_key_base_from_param_name=new_key_base_from_param_name,
        param_name=param_name
        )
    handle_registy_element_case = _internal_utils.make_maybe_list_case_handler(
        handle_single_case=handle_registry_element_case_single,
        allow_extended_prompt_for_single_case=True  # why not? it's a free feature at this point
    )  # this one is for either T, list[T], tuple[T], or Union[T, list[T], tuple[T]] where T is a registry element
    # also meant to handle varargs with type T
    # T may itself be a Union of types in the registry and None

    handle_callable_or_type_case = _internal_utils.make_maybe_list_case_handler(
        partial(get_callable_from_extended_prompt, default_module=default_module), 
        allow_extended_prompt_for_single_case=True
        )
    # this one is for Callable, type[T], Union[Callable, type[T]], list[Callable], list[type[T]], etc.  where T is some class (not necessarily in the registry)
    # also meant to handle varargs with type Callable, type[T], or Union[Callable, type[T]]

    handle_callable_or_registry_element_case_single = partial(
        get_callable_or_prep_class_from_extended_prompt,
        config=config,
        default_module=default_module,
        registry=registry,
        keys=keys,
        current_key=current_key,
        new_key_postfix=new_key_postfix,
        new_key_body= param_name if new_key_base_from_param_name else None,
        new_key_base_from_param_name=new_key_base_from_param_name,
        param_name=param_name
    )
    handle_callable_or_registry_element_case = _internal_utils.make_maybe_list_case_handler(handle_callable_or_registry_element_case_single, allow_extended_prompt_for_single_case=True)
    # this one is for Union[Callable, T], list[Union[Callable, T]], tuple[Union[Callable, T]], and Union[Callable, T, list[Union[Callable, T]], tuple[Union[Callable, T]]] where T is a registry element
    # (think cases where some model needs either a pure function as activation function, or some torch.nn.Module or something like that)

    # Now we define the checks for the various cases
    # first we do the simple checks
    registry_element_check = partial(is_in_registry, registry=registry)
    callable_or_type_check = _internal_utils.annotation_is_callable_or_type
    callable_or_registry_element_check = partial(_internal_utils.annotation_is_registry_element_or_callable, registry=registry)

    # we now lift these type checks to allow for unions of their positive types (and None) and lists/tuples of these unions
    # these lifted versions return an auxiliary boolean indicating whether the annotation includes lists or tuples
    registry_element_check = _internal_utils.make_general_type_check(registry_element_check, allow_none=True)
    callable_or_type_check = _internal_utils.make_general_type_check(callable_or_type_check, allow_none=True)
    callable_or_registry_element_check = _internal_utils.make_general_type_check(callable_or_registry_element_check, allow_none=True)

    # now we do the work
    annotation = details.annotation
    is_var_positional = details.kind is details.VAR_POSITIONAL

    # the following code is ugly for a reason
    # we really do need to know whether the annotation allows for list/tuples or not
    # because if it doesn't, we don't need to check whether contents is a valid prompt/extended prompt
    # and checking if it is a valid prompt/extended prompt leads to user warnings because it is hard to know for sure whether (str, str) is meant to be a prompt or a tuple of two prompts. 
    
    # also, we can't use match statements because we want to support python 3.9
    if (check_result := registry_element_check(annotation))[0]:  # first element is whether it's the right type, second is whether the annotation includes lists or tuples
        return handle_registy_element_case(contents, allow_multiple=(is_var_positional or check_result[1]))
    elif (check_result := callable_or_type_check(annotation))[0]:
        return handle_callable_or_type_case(contents, allow_multiple=(is_var_positional or check_result[1]))
    elif (check_result := callable_or_registry_element_check(annotation))[0]:  # NB the order is important here, this one would give True if any of the above give True
        return handle_callable_or_registry_element_case(contents, allow_multiple=(is_var_positional or check_result[1]))
    else:
        # we assume we're dealing with a simple type such as int, str, etc.
        return contents 

def get_callable_from_extended_prompt(extended_prompt: ExtendedPrompt, default_module:Union[None, ModuleType, str]=None):
    """get_callable_from_extended_prompt 
    return a callable object specified by an ExtendedPrompt

    :param extended_prompt: one of:
        path, name, local_config  
            where name should refer to a callable in the module located at path
        name, local_config  
            where name should refer to a callable in default_module
        callable, local_config
        path, name
        name
        callable
        None
    :return: the specified callable. 
    If a local_config is provided, the callable is wrapped in a partial with that local_config
    NB if extended_prompt is None, None is returned
    """
    prompt, local_config = split_extended_prompt(extended_prompt)
    callable_object, _ = process_prompt(prompt, default_module=default_module)
    if local_config:
        callable_object = partial(callable_object, **local_config)
    return callable_object
    
def get_callable_or_prep_class_from_extended_prompt(
        extended_prompt: ExtendedPrompt, 
        config: dict, 
        default_module:Union[None, ModuleType, str]=None, 
        registry:list=type_registry, 
        keys:Union[None, list]=None, 
        current_key:Union[None, str, tuple[str]]=None, 
        new_key_postfix:str='_config', 
        new_key_body:Union[None, str]=None, 
        new_key_base_from_param_name:bool = False, 
        ignore_params:Union[None, tuple[str], list[str]] = None,
        param_name:Union[None, str]=None
        ):
    """
    Check whether the extended_prompt points to a registry element
    if it does, continue with prep_class_from_extended_prompt
    if it doesn't, continue with get_callable_from_extended_prompt
    """
    # the alternative check would be to check whether the prompt points to a callable or a type
    # the downside of doing that is that one might use meta-classes to create classes that behave like functions
    # if that's the case, we'd want to treat them as callables, not as classes
    
    # the downside is that if the user provides a class of a type that is not in the registry, we'll treat it as a callable
    # but then again, adding it to the registry is not that hard and shouldn't have any negative side effects
    
    # this function results in two calls to split_extended_prompt and process_prompt, but that shouldn't be a problem
    prompt, _ = split_extended_prompt(extended_prompt)
    callable_object, _ = process_prompt(prompt, default_module=default_module)
    if is_in_registry(callable_object, registry=registry):
        return prep_class_from_extended_prompt(
            extended_prompt=extended_prompt,
            config=config,
            default_module=default_module,
            registry=registry,
            keys=keys,
            current_key=current_key,
            new_key_postfix=new_key_postfix,
            new_key_body=new_key_body,
            new_key_base_from_param_name=new_key_base_from_param_name,
            ignore_params=ignore_params,
            param_name=param_name,
        )
    return get_callable_from_extended_prompt(extended_prompt, default_module=default_module)


def match_signature_from_config(
        signature: inspect.Signature,
        config: dict,
        keys: Union[None, list],
        registry:list = type_registry,
        default_module:Union[None, ModuleType, str] = None,
        new_key_postfix:str = '_config',
        new_key_base_from_param_name:bool = False,
        ignore_params:Union[None, tuple[str], list[str]] = None,
        )->tuple[dict, list]:
    """ 
    :param signature: the call signature to be matched
    :param config: the config to pull the values from
    :param keys: (optional) potential children of config where values may be retrieved from. 
        Righter most is processed last.
        If B is processed after A, B can overrule the values in A. 
        A key can either be a string or a tuple of strings
        in case of a tuple of strings, it's treated as config[key[0]][key[1]]...
    :param registry: list of types that require initialization (or just calling) based on config before being returned
    :param default_module: optionally the default module from which to get classes if the prompt only specifies a class name
        either an actual module, 
        or a path to a module
    :param new_key_postfix: string used for finding names of subconfigs 
    :param new_key_base_from_param_name: if True, uses the parameter name to look for sub-configs instead of the class name
    :param ignore_params: optional collection of parameter names to be ignored in the process (will be added to missing_args)
    :returns: a dictionary with the relevant keyword-value pairs for the signature pulled from config and a list of missing arguments
        NB any classes in registry remain un-initialized PostponedInitialization instances that can be realized by calling the initialize() method
    """
    keys = keys or []
    # print(f'handeling {signature=}')
    ignore_params = ignore_params or ()

    # first we get all the default kwargs from the signature
    kwargs = {
        param_name: param.default 
        for param_name, param in signature.parameters.items() if param_name not in ignore_params
        if param.default is not param.empty 
    }

    # now we update the kwargs based on the config
    kwargs.update(
        _get_kwargs_update(
            signature=signature, 
            config=config, 
            registry=registry, 
            default_module=default_module, 
            keys=keys, 
            new_key_postfix=new_key_postfix,
            new_key_base_from_param_name=new_key_base_from_param_name,
            ignore_params=ignore_params
            )
    )

    # now update further based on keys
    # print(f'before handling keys: {kwargs=}')
    # print(f'updating based on {keys=}')
    for key in keys:
        if has_index(config, key):
            sub_config = get_from_index(config, key)
            #sub_keys = _get_sub_keys(keys, key)
            # print(f"getting kwarg update for {key}")
            kwargs.update(
                _get_kwargs_update(
                    signature=signature, 
                    config=config, 
                    registry=registry, 
                    default_module=default_module, 
                    keys=keys, 
                    sub_config=sub_config, 
                    current_key=key, 
                    new_key_postfix=new_key_postfix,
                    new_key_base_from_param_name=new_key_base_from_param_name,
                    ignore_params=ignore_params
                    )
            )
            # print(f"after handling {key=}, {kwargs=}")

    # find out what arguments are still missing
    missing_args = [
        name for name, parameter in signature.parameters.items() 
        if name not in kwargs and parameter.kind is not parameter.VAR_KEYWORD
        ]
    # print(f'finished handling {signature=}')

    return kwargs, missing_args


def _get_kwargs_update(
        signature, 
        config, 
        registry, 
        default_module, 
        keys, 
        sub_config=None, 
        current_key=None, 
        new_key_postfix='_config',
        new_key_base_from_param_name=False,
        ignore_params=None,
        ): # TODO write docstring
    ignore_params = ignore_params or ()
    return {
        param_name: get_parameter_from_config(
            param_name=param_name, 
            details=parameter, 
            config=config, 
            registry=registry, 
            default_module=default_module, 
            keys=keys, 
            sub_config=sub_config, 
            current_key=current_key, 
            new_key_postfix=new_key_postfix,
            new_key_base_from_param_name=new_key_base_from_param_name
            ) 
        for param_name, parameter in signature.parameters.items()
        if param_name in (sub_config if sub_config is not None else config)
        and param_name not in ignore_params
    }


def get_model_from_config(
        config: dict, 
        model_prompt:str="model_type", 
        default_module_key:str="architecture", 
        registry:list=type_registry, 
        keys:Union[None, list]=None, 
        missing_kwargs:Union[None, dict]=None,
        sub_config_postfix:str = '_config',
        sub_config_from_param_name:bool = True,
        model_sub_config_name_base:Union[None, str]=None,
        add_model_module_to_architecture_default_module:bool=False,
        additional_architecture_default_modules:Union[None, tuple[ModuleType], list[ModuleType], ModuleType]=None,
        initialize:bool=True,
        )->object:
    """ 
    produce a model from a config dict
    :param config: config dict specifying the model
    :param model_prompt: key in config. prompt=config[model_prompt] should point to where the model class is located 
    :param default_module_key: optional key in config pointing to a module containing architectures (e.g. for encoders, decoders, etc.)
        if not None, config[default_module_key] should be a path to a python file
    :param registry: list of types that require initialization (or just calling) based on config before being returned
        defaults to common_dl_utils.type_registry.type_registry
    :param keys: optional list of keys in config that contain sub-configs
        Righter most is processed last.
        If B is processed after A, B can overrule the values in A. 
        A key can either be a string or a tuple of strings
        in case of a tuple of strings, it's treated as config[key[0]][key[1]]...
    :param missing_kwargs: optional dict of keyword arguments to be passed to the model class (can overrule the values specified in config)
    :param sub_config_postfix: string used for finding names of subconfigs 
    :param sub_config_from_param_name: if True, uses the parameter name to look for sub-configs, otherwise uses the class name
    :param model_sub_config_name_base: optional string to use for finding the sub-config for this model
        if provided, will look for f'{model_sub_config_name_base}{sub_config_postfix}'
        otherwise, will look for f'{class_name}{sub_config_postfix}', where class_name is the class name specified in the model prompt in the config.
    :param add_model_module_to_architecture_default_module: if True, will add the model module to the default module specified by default_module_key
        NB in this case, the model_prompt should be a tuple or list of length 2, where the first element is the path to the model module
    :param additional_architecture_default_modules: optional additional modules to be added to the default module specified by default_module_key
    :param initialize: if True, initialize the model before returning it, otherwise return a PostponedInitialization instance
    :returns: an initialized model as specified by config and missing_kwargs if initialize is True, else a PostponedInitialization instance
    """
    missing_kwargs = missing_kwargs or {}
    prompt = config[model_prompt]
    default_module = load_from_path(name="architecture", path=config[default_module_key]) if default_module_key is not None else None
    if add_model_module_to_architecture_default_module:
        if not isinstance(prompt, (tuple, list)):
            raise ValueError(f"if {add_model_module_to_architecture_default_module=} is True, {model_prompt=} should be a tuple or list of length 2 so as to specify the path of the model module")
        model_module_path = prompt[0]
        model_module = load_from_path(name="model_module", path=model_module_path)
        default_module = MultiModule(default_module, model_module) if default_module is not None else model_module
    if additional_architecture_default_modules is not None:
        if isinstance(additional_architecture_default_modules, (tuple, list)):
            default_module = MultiModule(default_module, *additional_architecture_default_modules)
        else: 
            default_module = MultiModule(default_module, additional_architecture_default_modules)
    uninitialized_model = prep_class_from_config(
        prompt=prompt, 
        config=config, 
        default_module=default_module, 
        registry=registry, 
        keys=keys, 
        new_key_postfix=sub_config_postfix,
        new_key_body=model_sub_config_name_base,
        new_key_base_from_param_name=sub_config_from_param_name
        )
    if missing_kwargs:
        uninitialized_model.resolve_missing_args(missing_kwargs)
    if initialize:
        return uninitialized_model.initialize()
    else:
        return uninitialized_model
