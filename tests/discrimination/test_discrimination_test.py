"""Tests for the discrimination test."""

from __future__ import annotations

import pytest

from sensopy import DiscriminationTest
from sensopy.discrimination import (
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
from sensopy.discrimination.methods import DiscriminationMethod


@pytest.mark.parametrize(
    "method,correct,panelists",
    [
        (TriangleMethod(), 19, 30),
        (TwoAFCMethod(), 19, 30),
        (ThreeAFCMethod(), 19, 30),
        (FourAFCMethod(), 19, 30),
        (MultipleAFCMethod(10), 19, 30),
        (SpecifiedTetradMethod(), 19, 30),
        (UnspecifiedTetrad(), 19, 30),
        (DualPairMethod(), 19, 30),
        (DuoTrioMethod(), 19, 30),
        (pytest.param(MPlusNMethod(4, 3), 19, 30, marks=pytest.mark.slow)),
        (
            pytest.param(
                MPlusNMethod(4, 3, specified=True),
                19,
                30,
                marks=pytest.mark.slow,
            )
        ),
        (pytest.param(MPlusNMethod(2, 2), 19, 30, marks=pytest.mark.slow)),
        (pytest.param(MPlusNMethod(3, 3), 19, 30, marks=pytest.mark.slow)),
    ],
    ids=[
        "triangle",
        "two_afc",
        "three_afc",
        "four_afc",
        "m_afc",
        "stetrad",
        "utetrad",
        "dualpair",
        "duotrio",
        "mplusn(M=4, N=3, Unspecified)",
        "mplusn(M=4, N=3, Specified)",
        "mplusn(M=2, N=2, Unspecified)",
        "mplusn(M=3, N=3, Unspecified)",
    ],
)
def tests_discrimination(
    method: DiscriminationMethod,
    correct: int,
    panelists: int,
) -> None:
    """Test the discrimination protocols."""
    test = DiscriminationTest(method)

    t1 = test.difference(correct, panelists)
    assert t1.pc.estimate == pytest.approx(correct / panelists)
    # assert t1.p_value < t1.alpha

    t2 = test.equivalence(correct, panelists)
    assert t2.pc.estimate == pytest.approx(correct / panelists)
    # assert t2.p_value > 0.95
