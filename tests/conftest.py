"""Fixtures and hooks for pytest."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy

if TYPE_CHECKING:
    import pytest


def pytest_report_header(config: pytest.Config) -> list[str]:  # noqa: ARG001
    """Return a list of strings to be displayed in the header of the report."""
    return [
        f"SciPy version: {scipy.__version__}",
        f"NumPy version: {np.__version__}",
    ]
