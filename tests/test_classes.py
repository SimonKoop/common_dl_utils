from common_dl_utils.type_registry import register_type
from dataclasses import dataclass
from typing import Callable, Union

@register_type
@dataclass
class BaseTestType:
    pass 

@dataclass
class TestType1(BaseTestType):
    """ 
    Type for testing whether the building of models from configs works correctly
    """

    param_1: int 
    param_2: int
    toggle_1: bool
    toggle_2: bool

@dataclass
class TestType2(BaseTestType):
    """ 
    Another type for testing whether the building of models from configs works correctly
    """
    param_1: int 
    param_2: int
    toggle_1: bool 
    sub_model_tt2: BaseTestType  

@dataclass
class TestType3(BaseTestType):
    "Third one"
    param_1: int 
    param_2: int 
    toggle_1: bool 
    toggle_2: bool
    sub_model_tt3: BaseTestType  # NB if these sub_model parameters have the same name, you can very easily get into infinite recursions

@register_type
class TestType4:
    def __init__(self, *elements:BaseTestType, some_func:Callable):
        self.elements = elements
        self.some_func = some_func
    def __str__(self):
        return f"TestType4(elements={self.elements}, some_func={self.some_func.__name__})"
    def __repr__(self):
        return str(self)

@register_type
class TestType5:
    def __init__(
            self, 
            a:Union[TestType1, TestType2], 
            b:Union[BaseTestType, list[BaseTestType], None], 
            c:Union[type[BaseTestType], list[type[BaseTestType]], None], 
            d: Union[BaseTestType, Callable, list[BaseTestType], list[Callable], list[Union[BaseTestType, Callable]]]
            ):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __str__(self):
        return f"TestType5(a={str(self.a)}, b={str(self.b)}, c={str(self.c)}, d={str(self.d)})"
    def __repr__(self):
        return str(self)
