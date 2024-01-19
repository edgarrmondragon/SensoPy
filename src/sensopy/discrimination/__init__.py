"""Discrimination tests."""

from __future__ import annotations

from .discrimination import DiscriminationTest, Statistic, TestResults
from .methods import (
    DualPairMethod,
    DuoTrioMethod,
    FourAFCMethod,
    MPlusNMethod,
    MultipleAFCMethod,
    SpecifiedTetradMethod,
    ThreeAFCMethod,
    TriangleMethod,
    TwoAFCMethod,
    UnspecifiedTetrad,
)

__all__ = [
    "DiscriminationTest",
    "Statistic",
    "TestResults",
    "DualPairMethod",
    "DuoTrioMethod",
    "FourAFCMethod",
    "MPlusNMethod",
    "MultipleAFCMethod",
    "SpecifiedTetradMethod",
    "ThreeAFCMethod",
    "TriangleMethod",
    "TwoAFCMethod",
    "UnspecifiedTetrad",
]
