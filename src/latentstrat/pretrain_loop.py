"""Deprecated V6.1 module alias for prior training helpers."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("latentstrat.pretraining.prior.train")
