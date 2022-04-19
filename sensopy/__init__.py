"""Top-level package for SensoPy."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 8):
    from importlib.metadata import version
else:
    from importlib_metadata import version

__version__ = version(__name__)
"""Package version"""
