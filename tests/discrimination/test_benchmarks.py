"""Benchmarks for the discrimination test, run with `pytest --codspeed`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sensopy.discrimination import (
    SPECIFIED_TETRAD,
    THREE_AFC,
    TRIANGLE,
    TWO_AFC,
    DiscriminationTest,
    MPlusNMethod,
)

if TYPE_CHECKING:
    from pytest_codspeed import BenchmarkFixture


METHODS = {
    "triangle": TRIANGLE,
    "two_afc": TWO_AFC,
    "three_afc": THREE_AFC,
    "stetrad": SPECIFIED_TETRAD,
    "mplusn": MPlusNMethod(4, 3, seed=0),
}

# (correct, panelists): a typical result, plus the edges where the confidence
# limits are clamped to the guessing probability and to 1.
CASES = {
    "typical": (19, 30),
    "at_guessing": (0, 30),
    "all_correct": (30, 30),
}


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", METHODS)
def test_benchmark_difference(benchmark: BenchmarkFixture, method: str, case: str) -> None:
    """Benchmark a difference test, which solves for d' and its confidence limits."""
    test = DiscriminationTest(METHODS[method])
    correct, panelists = CASES[case]
    benchmark(test.difference, correct, panelists)


@pytest.mark.parametrize("method", METHODS)
def test_benchmark_equivalence(benchmark: BenchmarkFixture, method: str) -> None:
    """Benchmark an equivalence test."""
    test = DiscriminationTest(METHODS[method])
    benchmark(test.equivalence, 19, 30)
