from common_dl_utils.type_registry import register_type
from dataclasses import dataclass

@register_type
@dataclass
class BaseTestType:
    pass 

@register_type
@dataclass
class TestType1(BaseTestType):
    """ 
    Type for testing whether the building of models from configs works correctly
    """

    param_1: int 
    param_2: int
    toggle_1: bool
    toggle_2: bool

@register_type
@dataclass
class TestType2(BaseTestType):
    """ 
    Another type for testing whether the building of models from configs works correctly
    """
    param_1: int 
    param_2: int
    toggle_1: bool 
    sub_model_tt2: BaseTestType  

@register_type
@dataclass
class TestType3(BaseTestType):
    "Third one"
    param_1: int 
    param_2: int 
    toggle_1: bool 
    toggle_2: bool
    sub_model_tt3: BaseTestType  # NB if these sub_model parameters have the same name, you can very easily get into infinite recursions
