import importlib
import os
import types
import sys

__all__ = [
    "load_from_path",
    "make_full_path",
    "get_module_from_config",
    "MultiModule"
]

def load_from_path_old(name, path):
    """ 
    Deprecated, use load_from_path instead
    load a module from a path

    :param name: name to be given to the module
    :param path: path to the module that is to be loaded
    :return: a loaded module

    NB if a module with the same name is already loaded, this will modify that module
    Objects in the name space of the new module will be added to the existing module
    This can result in weird bugs
    """
    return importlib.machinery.SourceFileLoader(name, make_full_path(path)).load_module()


def load_from_path(name, path, register_in_sys_modules=False):
    """load_from_path load a module from a path

    :param name: name to be given to the module
    :param path: path to the module that is to be loaded (can be either absolute or relative)
    :param register_in_sys_modules: whether to add the resulting module to sys.modules under name, defaults to False
    :return: a loaded module

    Implementation from https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly

    NB contrary to load_from_path_old, this does not modify existing modules that are loaded under the same name
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if register_in_sys_modules:
        sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def make_full_path(path):
    """make a potentially relative path a full path"""
    if path[0] == "./":
        return f"{os.getcwd()}/{path[2:]}"
    return path


def get_module_from_config(config, key, default, name=None):
    """
    Load a module from config[key] if config contains key,
    otherwise use default

    :param config: a mapping
    :param key: key in mapping that should potentially point to a path of a module
    :param default: default module or path to resort to if key is not present in config
    :param name: (optional) name to use for load_from_path(name, path). If None, key will be used
    """
    if key not in config:
        if isinstance(default, types.ModuleType):
            return default
        path = default
    else:
        path = config[key]

    if name is None:
        name = key

    return load_from_path(name, path)

class MultiModule(types.ModuleType):
    """ 
    Reference holder for multiple modules
    Mostly meant to enable the use of multiple default modules in config_realization
    Modules will be searched through in order from left to right
    """
    def __init__(self, *modules):
        self._sub_modules = list(modules)
        self.__name__ = f"MultiModule({', '.join([m.__name__ for m in modules])})"
        self.__package__ = ''
        self.__loader__ = None
        self.__spec__ = importlib.machinery.ModuleSpec(
            self.__name__,
            None,
        )


    def __getattr__(self, name):
        for module in self._sub_modules:
            try:
                attr = getattr(module, name)
                return attr 
            except AttributeError:
                pass
        raise AttributeError(f"None of the modules in {[module.__name__ for module in self._sub_modules]} has an attribute {name}")
