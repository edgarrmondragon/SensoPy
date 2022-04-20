"""Tests for the discrimination methods."""

from __future__ import annotations

import pytest

from sensopy.discrimination.methods import DiscriminationMethod


def test_abstract_psychometric_function():
    """Test abstract psychometric function."""

    class _CustomGuessing(DiscriminationMethod):
        @property
        def guessing(self) -> float:
            return 1 / 2

    with pytest.raises(TypeError, match="abstract method"):
        _CustomGuessing()


def test_abstract_guessing():
    """Test abstract guessing property."""

    class _CustomPsychometric(DiscriminationMethod):
        def psychometric_function(self, d):
            return 0.5

    with pytest.raises(TypeError, match="abstract method"):
        _CustomPsychometric()


def test_discriminator():
    """Test discriminator."""

    class _CustomDiscriminator(DiscriminationMethod):
        def psychometric_function(self, d):
            return d

        @property
        def guessing(self):
            return 0.25

    method = _CustomDiscriminator()
    assert method.discriminators(0.5) == pytest.approx(1 / 3)
