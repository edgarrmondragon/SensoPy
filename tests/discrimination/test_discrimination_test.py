"""Tests for the discrimination test."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.stats import binom

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


@pytest.mark.parametrize(
    ("x", "n", "pd0"),
    [
        (20, 60, 0.5),
        (12, 40, 0.3),
        (30, 100, 0.4),
    ],
)
def test_equivalence_power_is_exact(x: int, n: int, pd0: float) -> None:
    """Equivalence power equals the probability of landing in the rejection region."""
    alpha = 0.05
    pg = TRIANGLE.guessing
    pc0 = pg + (1 - pg) * pd0
    pc = x / n
    ks = np.arange(n + 1)
    reject = binom.cdf(ks, n, pc0) <= alpha
    exact = float(binom.pmf(ks, n, pc)[reject].sum())
    assert DiscriminationTest(TRIANGLE).equivalence(x, n, pd0=pd0).power == pytest.approx(exact)


def test_equivalence_power_alpha_on_tail_boundary() -> None:
    """When alpha equals a tail probability exactly, that count is in the rejection region."""
    # 2-AFC, n=10, pc0=0.5: P(X <= 2) = 56/1024 = 0.0546875, exactly representable
    n, x = 10, 4
    conf_level = 1 - 56 / 1024
    assert binom.cdf(2, n, 0.5) == 1 - conf_level  # the ``xcrit`` adjustment is not needed

    res = DiscriminationTest(TWO_AFC).equivalence(x, n, pd0=0, conf_level=conf_level)
    assert res.power == pytest.approx(binom.cdf(2, n, x / n))
