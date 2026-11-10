"""Tests for the discrimination methods."""

from __future__ import annotations

import pytest

from sensopy.discrimination.methods import (
    THREE_AFC,
    TRIANGLE,
    TWO_AFC,
    DiscriminationMethod,
)
from sensopy.discrimination.mplusn import mplusn_mc


def test_abstract_psychometric_function() -> None:
    """Test abstract psychometric function."""

    class _CustomGuessing(DiscriminationMethod):
        @property
        def guessing(self) -> float:
            return 1 / 2

    with pytest.raises(TypeError, match="abstract method"):
        _CustomGuessing()  # type: ignore[abstract]  # ty: ignore[abstract-class-instantiation]


def test_abstract_guessing() -> None:
    """Test abstract guessing property."""

    class _CustomPsychometric(DiscriminationMethod):
        def psychometric_function(self, d: float) -> float:  # ruff: ignore[unused-method-argument]
            return 0.5

    with pytest.raises(TypeError, match="abstract method"):
        _CustomPsychometric()  # type: ignore[abstract]  # ty: ignore[abstract-class-instantiation]


def test_discriminator() -> None:
    """Test discriminator."""

    class _CustomDiscriminator(DiscriminationMethod):
        def psychometric_function(self, d: float) -> float:
            return d

        @property
        def guessing(self) -> float:
            return 0.25

    method = _CustomDiscriminator()
    assert method.discriminators(0.5) == pytest.approx(1 / 3)


def test_invalid_m_plus_n() -> None:
    """Test invalid m plus n method."""
    with pytest.raises(
        ValueError,
        match=r"Invalid combination of parameters. M >= N expected.",
    ):
        mplusn_mc(1, 3)


def test_simulate_scalar() -> None:
    """A single simulation returns an int within [0, n]."""
    x = TRIANGLE.simulate(1.0, 50, rng=1)
    assert isinstance(x, int)
    assert 0 <= x <= 50


def test_simulate_reproducible() -> None:
    """The same seed gives the same results."""
    a = THREE_AFC.simulate(1.0, 30, size=5, rng=42)
    b = THREE_AFC.simulate(1.0, 30, size=5, rng=42)
    assert a.tolist() == b.tolist()


def test_simulate_mean() -> None:
    """The mean number of correct responses approaches n * P_c."""
    n = 100
    thurstone = 0.8
    xs = TWO_AFC.simulate(thurstone, n, size=20_000, rng=0)
    expected = n * TWO_AFC.psychometric_function(thurstone)
    assert xs.mean() == pytest.approx(expected, abs=0.2)


def test_simulate_null() -> None:
    """With d' = 0 the success rate is the guessing probability."""
    xs = TRIANGLE.simulate(0.0, 90, size=20_000, rng=0)
    assert xs.mean() == pytest.approx(90 * TRIANGLE.guessing, abs=0.2)


@pytest.mark.parametrize(("d", "n"), [(-1.0, 10), (1.0, 0)])
def test_simulate_invalid(d: float, n: int) -> None:
    """Invalid arguments raise ValueError."""
    with pytest.raises(ValueError, match="must be"):
        TRIANGLE.simulate(d, n)
