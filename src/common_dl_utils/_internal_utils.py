import os
import warnings
from typing import Any, Union, get_origin, get_args, Callable, Mapping
from common_dl_utils.type_registry import is_in_registry, type_registry
import functools
import inspect

NoneType = type(None)

def annotation_is_callable_or_type(annotation):
    """
    is the annotation either Callable or type[something]?
    (NB this excludes Unions of Callables and types)
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
        return True
    return annotation in (callable, Callable, 'callable', 'Callable', 'type')  # maybe remove string checks as we don't support them anywhere else 

def annotation_is_registry_element_or_callable(annotation, registry=type_registry):
    return is_in_registry(annotation, registry=registry) or (annotation in (callable, Callable, 'callable', 'Callable'))

def make_union_type_check(type_check:Callable, allow_none=True, aggregate_auxiliaries=any):
    """make a type check for a Union of types, where type_check is a type check for the individual types

    :param type_check: a function that takes an annotation and returns whether it is of a type we're interested in
        this function may also return some auxiliary (boolean) information e.g. whether we may be dealing with a list of types
    :param allow_none: whether to allow None in the Union, defaults to True
    :param aggregate_auxiliaries: function ao aggregate the auxiliary information from the type checks in case of a Union, defaults to any
    :returns: a function that takes an annotation and returns whether it is a Union of types that satisfy type_check, and possibly some auxiliary information
    """
    def union_type_check(annotation):
        origin = get_origin(annotation)
        if origin != Union:
            return type_check(annotation) or (annotation in (NoneType, None) and allow_none)
        results = [type_check(arg) for arg in get_args(annotation)]
        if isinstance(results[0], bool):
            #  get_args returns NoneType instead of None for Union[None, ...]
            return all(check or (arg is NoneType and allow_none) for check, arg in zip(results, get_args(annotation)))
        # type_check has returned some auxiliary information
        result, *auxiliaries = zip(*results)
        return (all(check or (arg is NoneType and allow_none) for check, arg in zip(result, get_args(annotation))),) + tuple(map(aggregate_auxiliaries, auxiliaries))
    return union_type_check

def make_list_or_tuple_type_check(type_check:Callable):
    def list_or_tuple_type_check(annotation):
        """check if the annotation is either list[T], tuple[T], or T, where type_check(T) is True
        Additionally check whether the annotation is actually list/tuple or not

        :param annotation: _description_
        :return: _description_
        """
        origin = get_origin(annotation)
        if origin not in (list, tuple):
            return type_check(annotation), False
        # if we get here, the annotation is a list or tuple
        return all(type_check(arg) for arg in get_args(annotation)), True
    return list_or_tuple_type_check

def make_general_type_check(type_check:Callable, allow_none=True, aggregate_auxiliaries=any):
    """
    make a general type check for a type that may be a Union, list, tuple, or just a single type for which type_check returns True
    :param type_check: a function that takes an annotation and returns whether it is of a type we're interested in
    :param allow_none: whether to include None in the allowed types, defaults to True
    :param aggregate_auxiliaries: function for aggregating the auxiliary information on whether the type includes lists or tuples, defaults to any
    :return: a function taking annotations and returning whether they are of the type we're interested in, and whether it includes lists or tuples
    """
    return make_union_type_check(
        make_list_or_tuple_type_check(
            make_union_type_check(
                type_check,
                allow_none=allow_none
            )
        ),
        allow_none=allow_none,
        aggregate_auxiliaries=aggregate_auxiliaries
    )  # the reason for doing it this way is that we might well encounter annotations like Union[T, S, List[Union[T, S]], Tuple[Union[T, S]], None]


def is_valid_extended_prompt(maybe_prompt:Any):
    """ 
    is the argument a valid extended prompt?
    """# TODO rewrite to decrease level of nesting
    # we want to support python 3.9, so we can't use match
    if maybe_prompt is None:
        return True
    elif callable(maybe_prompt):
        return True
    elif isinstance(maybe_prompt, str):
        return True
    # only options for extended prompts are now tuples and lists
    elif isinstance(maybe_prompt, (tuple, list)):
        if len(maybe_prompt) > 3 or len(maybe_prompt) < 2:
            return False # all extened prompt options are either two or three elements long
        if len(maybe_prompt) == 2:
            part_0, part_1 = maybe_prompt
            if isinstance(part_0, str) and isinstance(part_1, Mapping):
                return True
            elif isinstance(part_0, str):
                # string can only be followed by a string or a Mapping
                # and if we have two strings, the first one must be a valid path
                if not isinstance(part_1, str):
                    return False
                expected_outcome = os.path.exists(part_0)
                warnings.warn(
                    f"\nUpon checking whether {maybe_prompt=} is an extended prompt, "
                    f"we come to the conclusion that {expected_outcome=} based on whether {part_0} is a valid path. \n"
                    f"This check may result in weird bugs if {part_0} is a not a valid path only due to a typo, or is a valid path but not intended to be one."
                )
                # the reason this case is somewhat special is that _is_valid_extended_prompt is used primarily to check whether we are dealing with a single extended prompt or a list/tuple of extended prompts
                # so a list/tuple of two strings may either be an extended prompt (if the first one is a path) or a list/tuple of two prompts (both being just strings that shouldn't be paths)
                return expected_outcome
            else:
                return callable(part_0) and isinstance(part_1, Mapping)
        else: # len(maybe_prompt) == 3
            part_0, part_1, part_2 = maybe_prompt
            # no need to check if part_0 is a valid path in this case:
            # a Mapping can not be an extended prompt, so maybe_prompt can not be a list or tuple of extended prompts
            # and a typo in part_0 should result in a sensible down-stream ModuleNotFoundError
            return isinstance(part_0, str) and isinstance(part_1, str) and isinstance(part_2, Mapping)
    else:
        return False
    
def is_valid_prompt(maybe_prompt:Any):
    """
    is the argument a valid prompt?
    """
    if maybe_prompt is None:
        return True
    elif callable(maybe_prompt):
        return True
    elif isinstance(maybe_prompt, str):
        return True
    # only options for prompts are now tuples and lists of two strings
    if not isinstance(maybe_prompt, (tuple, list)):
        return False
    if len(maybe_prompt) != 2:
        return False
    part_0, part_1 = maybe_prompt
    if (not isinstance(part_0, str)) or (not isinstance(part_1, str)):
        return False
    expected_outcome = os.path.exists(part_0)
    warnings.warn(
        f"\nUpon checking whether {maybe_prompt=} is a prompt, "
        f"we come to the conclusion that {expected_outcome=} based on whether {part_0} is a valid path. \n"
        f"This check may result in weird bugs if {part_0} is a not a valid path only due to a typo, or is a valid path but not intended to be one."
    )
    # the reason this case is somewhat special is that _is_valid_extended_prompt is used primarily to check whether we are dealing with a single prompt or a list/tuple of prompts
    # so a list/tuple of two strings may either be a prompt (if the first one is a path) or a list/tuple of two prompts (both being just strings that shouldn't be paths)
    return expected_outcome
    
    
def map_partial(func, *args, **kwargs):
    """ 
    make a partial function out of func with args and kwargs, 
    and then return a function that maps this partial function over its arguments
    """
    partial_func = functools.partial(func, *args, **kwargs)
    def mapped_partial_function(mapped_arg, *mapped_args):
        return map(partial_func, mapped_arg, *mapped_args)
    return mapped_partial_function

def make_maybe_list_case_handler(handle_single_case, handle_multiple_case=None, allow_extended_prompt_for_single_case=True):
    prompt_check = is_valid_extended_prompt if allow_extended_prompt_for_single_case else is_valid_prompt
    if handle_multiple_case is None:
        def handle_multiple_case(contents):
            return type(contents)(handle_single_case(content) for content in contents)
    def handler(contents, allow_multiple:bool):
        if not allow_multiple:
            # we could still do the prompt_check and raise an Exception if prompt_check indicates that we are dealing with a list/tuple
            # but this might cause unwarranted Exceptions if we're dealing with multiple objects with the same parameter name, but a different annotation

            # more importantly, we don't want to do prompt_check if we don't need to because doing so would likely lead to a lot of warnings
            return handle_single_case(contents)
        if prompt_check(contents):
            return handle_single_case(contents)
        else:
            return handle_multiple_case(contents)
    return handler

def signature_to_var_keyword(func:Callable, signature=None):
    """
    wrap func so it can be called with **kwargs instead of requiring (var) positional arguments to be passed as such

    :param func: a callable to be wrapped
    :param signature: optional signature of said callable, defaults to None
        if None, the signature of func will be retrieved using inspect.signature
    :return: a wrapped version of func with signature **kwargs
    """
    if signature is None:
        signature = inspect.signature(func)

    positional_argument_names = [parameter_name for parameter_name, parameter in signature.parameters.items() if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.POSITIONAL_ONLY)]
    var_positional_argument_name = [parameter_name for parameter_name, parameter in signature.parameters.items() if parameter.kind == parameter.VAR_POSITIONAL]

    @functools.wraps(func)
    def wrapped_func(**kwargs):
        positional_arguments = [kwargs.pop(name) for name in positional_argument_names]
        if var_positional_argument_name:
            var_positional_argument = kwargs.pop(var_positional_argument_name[0], None)
            if var_positional_argument is not None and isinstance(var_positional_argument, (list, tuple)):
                positional_arguments += list(var_positional_argument)
            elif var_positional_argument is not None:
                positional_arguments.append(var_positional_argument)
        return func(*positional_arguments, **kwargs)
    wrapped_func.__doc__ = (wrapped_func.__doc__ or "") + "\n\nWrapped so as to accept only keyword arguments."
    return wrapped_func
