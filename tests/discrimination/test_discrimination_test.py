"""Tests for the discrimination test."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sensopy import DiscriminationTest
from sensopy.discrimination import (
    DUAL_PAIR,
    DUO_TRIO,
    FOUR_AFC,
    SPECIFIED_TETRAD,
    THREE_AFC,
    TRIANGLE,
    TWO_AFC,
    UNSPECIFIED_TETRAD,
    MPlusNMethod,
    MultipleAFCMethod,
)

if TYPE_CHECKING:
    from sensopy.discrimination.methods import DiscriminationMethod


@pytest.mark.parametrize(
    ("method", "correct", "panelists"),
    [
        (TRIANGLE, 19, 30),
        (TWO_AFC, 22, 30),  # 2-AFC has pg=1/2; 19/30 is not significant at a=0.05
        (THREE_AFC, 19, 30),
        (FOUR_AFC, 19, 30),
        (SPECIFIED_TETRAD, 19, 30),
        (UNSPECIFIED_TETRAD, 19, 30),
        (DUAL_PAIR, 22, 30),  # Dual Pair has pg=1/2; 19/30 is not significant at a=0.05
        (DUO_TRIO, 22, 30),  # Duo-Trio has pg=1/2; 19/30 is not significant at a=0.05
        (MultipleAFCMethod(10), 19, 30),
        (MPlusNMethod(4, 3, seed=0), 19, 30),
        (MPlusNMethod(4, 3, specified=True, seed=0), 19, 30),
        (MPlusNMethod(2, 2, seed=0), 19, 30),
        (MPlusNMethod(3, 3, seed=0), 19, 30),
    ],
    ids=[
        "triangle",
        "two_afc",
        "three_afc",
        "four_afc",
        "stetrad",
        "utetrad",
        "dualpair",
        "duotrio",
        "m_afc",
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
    assert t1.p_value < t1.alpha

    t2 = test.equivalence(correct, panelists)
    assert t2.pc.estimate == pytest.approx(correct / panelists)
    assert t2.p_value > 0.95
