"""Tests for the discrimination test."""

from __future__ import annotations

import pytest

from sensopy.discrimination import DiscriminationTest


@pytest.mark.parametrize(
    # "method,correct,panelists,alpha,power,p_value,d_prime,stderr,lower,upper",
    "method,method_kwargs,correct,panelists",
    [
        ("triangle", {}, 19, 30),
        ("two_afc", {}, 19, 30),
        ("three_afc", {}, 19, 30),
        ("four_afc", {}, 19, 30),
        ("duotrio", {}, 19, 30),
        ("m_afc", {"m": 10}, 19, 30),
        (pytest.param("mplusn", {"m": 4, "n": 3}, 19, 30, marks=pytest.mark.slow)),
        (pytest.param("mplusn", {"m": 2, "n": 2}, 19, 30, marks=pytest.mark.slow)),
        (pytest.param("mplusn", {"m": 3, "n": 3}, 19, 30, marks=pytest.mark.slow)),
    ],
    ids=[
        "triangle",
        "two_afc",
        "three_afc",
        "four_afc",
        "duotrio",
        "m_afc",
        "mplusn(4+3)",
        "mplusn(2+2)",
        "mplusn(3+3)",
    ],
)
def tests_discrimination(
    method: str,
    method_kwargs: dict,
    correct: int,
    panelists: int,
) -> None:
    """Test the discrimination protocols."""
    test = DiscriminationTest(method, **method_kwargs)

    t1 = test.difference(correct, panelists)
    assert t1.pg == pytest.approx(test.method.guessing)
    assert t1.pc.estimate == pytest.approx(correct / panelists)
    # assert t1.p_value < t1.alpha

    t2 = test.equivalence(correct, panelists)
    assert t2.pg == pytest.approx(test.method.guessing)
    assert t2.pc.estimate == pytest.approx(correct / panelists)
    # assert t2.p_value > 0.95
