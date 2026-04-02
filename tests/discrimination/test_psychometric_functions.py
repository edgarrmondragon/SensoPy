"""Tests for psychometric functions based on Bi (2015), Chapter 2."""

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
)

if TYPE_CHECKING:
    from sensopy.discrimination.methods import DiscriminationMethod


@pytest.mark.parametrize(
    ("method", "expected_pc"),
    [
        pytest.param(TWO_AFC, 0.7602, id="two_afc"),
        pytest.param(THREE_AFC, 0.6337, id="three_afc"),
        pytest.param(FOUR_AFC, 0.5520, id="four_afc"),
        pytest.param(DUO_TRIO, 0.5825, id="duotrio"),
        pytest.param(TRIANGLE, 0.4181, id="triangle"),
        pytest.param(UNSPECIFIED_TETRAD, 0.4938, id="utetrad"),
        pytest.param(SPECIFIED_TETRAD, 0.4625, id="stetrad"),
        pytest.param(DUAL_PAIR, 0.5733, id="dualpair"),
    ],
)
def test_psychometric_function_at_d1(
    method: DiscriminationMethod,
    expected_pc: float,
) -> None:
    """P_c at d'=1 matches Bi (2015), Table 2.13 (Example 2.4.3, p. 24)."""
    assert method.psychometric_function(1.0) == pytest.approx(expected_pc, abs=1e-3)


def test_3afc_d_prime_estimation() -> None:
    """Value of d' from 3-AFC with 63/100 correct matches Bi (2015), Example 2.4.1 (p. 19)."""
    result = DiscriminationTest(THREE_AFC).difference(63, 100)
    assert result.d_prime.estimate == pytest.approx(0.9872, abs=1e-3)
