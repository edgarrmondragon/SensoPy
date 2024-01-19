"""Top-level package for SensoPy."""

from __future__ import annotations

from importlib.metadata import version

from .discrimination import DiscriminationTest

__version__ = version(__name__)
"""Package version"""

__all__ = ["DiscriminationTest"]
