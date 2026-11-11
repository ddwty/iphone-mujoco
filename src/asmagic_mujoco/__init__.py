"""Deprecated import compatibility for the former package name.

All classes/submodules are the canonical iphone_mujoco objects, never copies.
"""
from importlib import import_module as _import_module
import sys as _sys
import warnings as _warnings

from iphone_mujoco import *
from iphone_mujoco import __version__, __all__

_warnings.warn("asmagic_mujoco is deprecated; use iphone_mujoco", DeprecationWarning, stacklevel=2)
for _name in ('camera', 'cli', 'configuration', 'control', 'devices', 'geometry', 'inspection', 'phone_lab', 'phone_scene', 'profiles', 'recording', 'robots', 'streams', 'virtual_device', 'wrist'):
    _module = _import_module("iphone_mujoco." + _name)
    _sys.modules[__name__ + "." + _name] = _module
    globals()[_name] = _module
del _name, _module, _import_module, _sys, _warnings
