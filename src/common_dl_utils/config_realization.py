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
from typing import Union, Callable, Any, get_origin
#from collections.abc import Sequence
from types import ModuleType
from common_dl_utils.type_registry import type_registry, contains
from common_dl_utils.module_loading import load_from_path, MultiModule
from common_dl_utils.trees import get_from_index, has_index, tree_map
from collections.abc import Iterable
from functools import partial

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
    """
    def __init__(self, cls:type, kwargs:dict, missing_args:Union[None, list]=None): 
        self.cls = cls 
        self.kwargs = kwargs 
        self.missing_args=missing_args
    
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
        return self.cls(**processed_self_kwargs)
    
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
        for arg_name in resolution.kwargs:
            if arg_name in self.missing_args:
                self.missing_args.remove(arg_name)
                self.kwargs[arg_name] = resolution.kwargs[arg_name]

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

    :returns: PostponedInitialization object

    NB In the documentation and in the naming, prompt is assumed to point to a class
    however, in principle, it can point to any callable, including functions
    """
    prompt, local_config = split_extended_prompt(extended_prompt)

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
        ignore_params=ignore_params
    )
    resolved_argument_names = list(postponed_init.kwargs.keys())
    # now if there are missing arguments, try to resolve them from the original config
    if postponed_init.missing_args:
        postponed_init.resolve_missing_args(
            resolution=prep_class_from_config(
                prompt=prompt,
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
    return postponed_init
    

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

    :returns: PostponedInitialization object

    NB In the documentation and in the naming, prompt is assumed to point to a class
    however, in principle, it can point to any callable, including functions
    """
    # first get the appropriate class from the appropriate module
    cls, class_name = process_prompt(prompt, default_module=default_module)
    # print(f"prepare {cls=} with {class_name=} from {prompt=}")
    if cls is None:
        return None

    new_key_body = class_name if new_key_body is None else new_key_body
    
    # get the call signature for cls
    signature = inspect.signature(cls)

    # update where to look for parameters in the config
    keys = list(keys) if keys is not None else []

    new_key_base = f'{new_key_body}{new_key_postfix}'
    # print(f"maybe add {new_key_base=} to {keys=}")
    maybe_add_key(new_key_base, keys, config)
    # print(f"now {keys=}")

    # if we're working in the context of a current_key, also add a multi-index based on it to the keys
    if isinstance(current_key, str):
        new_key = (current_key, new_key_base)
        maybe_add_key(new_key, keys, config)
    elif isinstance(current_key, tuple):
        for upto in range(1, len(current_key)+1):
            new_key = current_key[:upto] + (new_key_base, )
            maybe_add_key(new_key, keys, config)
    
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

    return PostponedInitialization(cls, kwargs, missing_args)
    
    
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
    # positional only arguments are not supported because we store all our arguments in a dict and feed that to the callable upon initialization/calling
    if details.kind is details.VAR_KEYWORD:
        raise NotImplementedError(f"We don't support matching VAR_KEYWORD arguments (raised in treating {param_name=} with {details=})")
    elif details.kind is details.POSITIONAL_ONLY:
        raise NotImplementedError(f"We don't support POSITIONAL_ONLY arguments (raised in treating {param_name=} with {details=})")
    
    contents = config[param_name] if sub_config is None else sub_config[param_name]
    if contains(details.annotation, registry=registry): 
        # print(f'handling {param_name=} and {details.annotation=} as a class')
        if details.kind is details.VAR_POSITIONAL:
            # contents should be a list of extended prompts
            # TODO add documentation on this to docstring
            contents = [
                prep_class_from_extended_prompt(
                    prompt=extended_prompt,
                    config=config,
                    default_module=default_module,
                    registry=registry,
                    keys=keys,
                    current_key=current_key,
                    new_key_postfix=new_key_postfix,
                    new_key_body= param_name if new_key_base_from_param_name else None,
                    new_key_base_from_param_name=new_key_base_from_param_name
                )
                for extended_prompt in contents
            ]
        else: 
            contents = prep_class_from_config(
                prompt=contents, 
                config=config, 
                default_module=default_module, 
                registry=registry, 
                keys=keys, 
                current_key=current_key, 
                new_key_postfix=new_key_postfix,
                new_key_body= param_name if new_key_base_from_param_name else None,
                new_key_base_from_param_name=new_key_base_from_param_name
                )
    elif _annotation_says_callable_or_type(details.annotation):
        # contents should be an ExtendedPrompt
        # case for this: activation functions
        return get_callable_from_extended_prompt(contents)
    return contents 

def _annotation_says_callable_or_type(annotation):
    """
    should the annotation be treated as saying that the parameter should be a callable or a type
    """ 
    annotation = get_origin(annotation) or annotation  
    # if the annotation is a parameterized generic, get the origin, 
    # otherwise just use the annotation
    if isinstance(annotation, type) and issubclass(annotation, type):
        # this is for cases like type[torch.nn.Module]
        # by using get_origin, we change this into type
        # and by checking if this is a subclass of type, we find out that indeed, the annotation is a type
        # note that other metaclasses (e.g. abc.ABCMeta) are also subclasses of type so we shoud indeed use issubclass
        # 
        # on the other hand if the annotation is e.g. torch.nn.Module, isinstance(annotation, type) will return True,
        # but according to the annotation this shouldn't be passed as a class but as an instance,
        # so we should return False
        #
        # the reason we need isinstance before issubclass is that if the annotation is a Union of types,
        # get_origin will return typing.Union. 
        # issubclass(typing.Union, type) will raise a TypeError, but isinstance(typing.Union, type) is False, 
        # so issubclass will never be called
        # TODO consider adding support for a union of types
        return True
    return annotation in (callable, Callable, 'callable', 'Callable', 'type')

def get_callable_from_extended_prompt(extended_prompt: ExtendedPrompt):
    """get_callable_from_extended_prompt 
    return a callable object specified by an extended_prompt

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
    """
    prompt, local_config = split_extended_prompt(extended_prompt)
    callable_object, _ = process_prompt(prompt)
    if local_config:
        callable_object = partial(callable_object, **local_config)
    return callable_object



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
    :returns: an initialized model as specified by config and missing_kwargs
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
    return uninitialized_model.initialize(**missing_kwargs)
