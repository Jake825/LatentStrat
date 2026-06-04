"""Deprecated V6.1 module alias for season training helpers."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("latentstrat.season.train")
