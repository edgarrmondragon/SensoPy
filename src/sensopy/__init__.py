"""Top-level package for SensoPy."""

from __future__ import annotations

from importlib.metadata import version

from .discrimination import DiscriminationTest  # noqa: F401

__version__ = version(__name__)
"""Package version"""
