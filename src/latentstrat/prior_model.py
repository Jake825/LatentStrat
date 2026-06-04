"""Deprecated V6.1 module alias for the prior model."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("latentstrat.pretraining.prior.model")
