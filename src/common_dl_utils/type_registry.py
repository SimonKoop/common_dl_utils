""" 
In order to automatically match configs to models, trainers, etc., it is useful to know
what parameters need instantiation. 
E.g. if we have 

>>> class MyModel(nn.Module):
>>>     def __init__(
>>>                 self, 
>>>                 some_sub_module: nn.Module
>>>                 some_parameter: int
>>>                 ):
>>>         ...

and 

>>> class MySubModule(nn.Module):
>>>     def __init__(
>>>                 self, 
>>>                 some_other_parameter: int
>>>                 ):
>>>         ... 

and we want to realize this from some config file
we want to be able to specify in the config file that some_sub_module will be an instance of MySubModule
that is initialized with the value of some_other_parameter also specified in the config.
But on the other hand, the some_parameter int should just be pulled from the config.

So this type registry is used to specify which types are initialized, and which don't need to be initialized
"""
from typing import Union, get_origin, get_args, ForwardRef
import warnings
from functools import wraps

def _deprecated(func=None, alternative=None):
    if func is None:
        return lambda func: _deprecated(func, alternative=alternative)
    warning = f"Call to deprecated function {func.__name__}."
    if alternative is not None:
        alt = alternative if isinstance(alternative, str) else alternative.__name__
        warning += f" Use {alt} instead."
    @wraps(func)
    def wrapper(*args, **kwargs):
        warnings.warn(warning, DeprecationWarning)
        return func(*args, **kwargs)
    return wrapper

__all__ = [
    "type_registry",
    "register_type",
    "contains",
    "is_subclass"
]

type_registry = []

def register_type(obj, registry=type_registry):
    registry.append(obj)
    return obj

def is_in_registry(maybe_element, registry=type_registry):
    return any(is_subclass(maybe_element, registry_element) for registry_element in registry)

@_deprecated(alternative=is_in_registry)
def contains(cls_spec, registry=type_registry):
    """contains: check if registry contains cls_spec
    True if cls_spec is either
    - a class that is a subclass of any element in the registry
    - a Union of types, of which at least one is a subclass of any element in the registry

    :param cls_spec: Either a class or a Union of classes
    :param registry: Collection of classes, defaults to type_registry
    :return: whether the cls_spec is considered to be in the registry
    """
    origin = get_origin(cls_spec)
    if origin is Union:
        # if cls_spec is Union[type_1, type_2, ...], check if any of these types are in the registry
        # the elements of the union can be found in typing.get_args(cls_spec) (of cls_spec.__args__)
        # This can likely be done in a nicer way in later Python versions, but in 3.9, you'll get a
        # TypeError: Subscripted generics cannot be used with class and instance checks
        return any(is_subclass(cls_spec_sub_type, registry_element) for cls_spec_sub_type in get_args(cls_spec) for registry_element in registry)
    elif origin is not None:
        # cls_spec is likely some parameterized type,
        # see if its origin is in the registry
        return contains(origin, registry)
    return any(is_subclass(cls_spec, registry_element) for registry_element in registry)
        
def is_subclass(maybe_subclass, maybe_parentclass):
    """is_subclass check if we should consider maybe_subclass a subclass of maybe_parentclass

    if maybe_subclass or maybe_parentclass are not type instances, they are not considered subclasses
    if maybe_subclass is a parameterized type, we check if its origin is a subclass of maybe_parentclass
        i.e. if maybe_subclass = T[V], we return issubclass(T, maybe_parentclass)
    """
    if not (isinstance(maybe_subclass, type) and isinstance(maybe_parentclass, type)):
        # if maybe_subclass is not a class, it is not a subclass
        # if maybe_parentclass is not a class, it is not a parentclass
        return False
    
    origin = get_origin(maybe_subclass)
    if origin is not None:
        # if maybe_subclass is a parameterized type, check if its origin is a subclass of maybe_parentclass
        return issubclass(origin, maybe_parentclass)
    return issubclass(maybe_subclass, maybe_parentclass)
