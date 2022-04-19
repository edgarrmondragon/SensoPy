"""Discrimination methods."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import scipy.special
from scipy.integrate import trapz
from scipy.stats import norm

from . import mplusn


class DiscriminationMethod(metaclass=ABCMeta):
    """A sensory discrimination method."""

    def __init__(self, name: str | None = None) -> None:
        """Initialize a discrimination method.

        Args:
            name: _description_. Defaults to None.
        """
        self.name = name

    @abstractmethod
    def psychometric_function(self, d: float) -> float:
        """Psychometric function.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            TODO.
        """
        ...

    @abstractproperty
    def guessing(self) -> float:
        """Guessing rate."""
        ...

    def discriminators(self, d: float) -> float:
        """Discriminators for the Dual Pair method.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            TODO.
        """
        pc = self.psychometric_function(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)


class Triangle(DiscriminationMethod):
    """Triangle method."""

    def __init__(self) -> None:
        """Initialize a triangle method."""
        super().__init__(name="triangle")

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the Triangle method.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            Probability of correct response.
        """
        delta = np.array([d])
        delta = delta.flatten()
        f1 = norm.pdf
        f2 = norm.cdf

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def _fi(z: np.ndarray) -> np.ndarray:
            x1 = -z * np.sqrt(3) + np.sqrt(2 / 3)
            x2 = -z * np.sqrt(3) - np.sqrt(2 / 3)
            return 2 * (f2(x1 * dr) + f2(x2 * dr)) * f1(z)

        x = np.linspace(0, 200, 10000)
        y = _fi(x)
        i = trapz(y, x)

        return i

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the Triangle method.
        """
        return 1 / 3


class TwoAFC(DiscriminationMethod):
    """Two-Alternative Forced-Choice method."""

    def __init__(self) -> None:
        """Initialize a 2-AFC discrimination method."""
        super().__init__(name="twoAFC")

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the 2-AFC method.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            Probability of correct response.
        """
        return norm.cdf(d / np.sqrt(2))

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the 2-AFC method.
        """
        return 1 / 2


class ThreeAFC(DiscriminationMethod):
    """Three-Alternative Forced-Choice method."""

    def __init__(self) -> None:
        """Initialize a 3-AFC discrimination method."""
        super().__init__(name="threeAFC")

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the 3-AFC method.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            Probability of correct response.
        """
        delta = np.array([d])
        delta = delta.flatten()

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def _fi(z: np.ndarray):
            return (norm.cdf(z) ** 2) * norm.pdf(z - dr)

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapz(y, x)

        return i

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the 3-AFC method.
        """
        return 1 / 3


class FourAFC(DiscriminationMethod):
    """Four-Alternative Forced-Choice method."""

    def __init__(self) -> None:
        """Initialize a 4-AFC discrimination method."""
        super().__init__(name="fourAFC")

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the 4-AFC method.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            Probability of correct response.
        """
        delta = np.array([d])
        delta = delta.flatten()

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def _fi(z: np.ndarray) -> np.ndarray:
            return (norm.cdf(z) ** 3) * norm.pdf(z - dr)

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapz(y, x)

        return i

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the 4-AFC method.
        """
        return 1 / 4


class MAFC(DiscriminationMethod):
    """m-Alternative Forced-Choice method."""

    def __init__(self, m: int) -> None:
        """Initialize a m-AFC discrimination method.

        Args:
            m: TODO.
        """
        super().__init__(name=f"{m}AFC")
        self.m = m

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the m-AFC method.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            Probability of correct response.
        """
        delta = np.array([d])
        delta = delta.flatten()

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def _fi(z: np.ndarray) -> np.ndarray:
            return (norm.cdf(z) ** (self.m - 1)) * norm.pdf(z - dr)

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapz(y, x)

        return i

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the m-AFC method.
        """
        return 1 / self.m


class SpecifiedTetrad(DiscriminationMethod):
    """Specified Tetrad method."""

    def __init__(self) -> None:
        """Initialize a specified tetrad discrimination method."""
        super().__init__(name="stetrad")

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the Specified Tetrad method.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            Probability of correct response.
        """
        delta = np.array([d])
        delta = delta.flatten()
        f1 = norm.pdf
        f2 = norm.cdf

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def _fi(z: np.ndarray) -> np.ndarray:
            return 2 * (f1(z) * f2(z) * (2 * f2(z - dr) - f2(z - dr) ** 2))

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapz(y, x)

        return 1 - i

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the Specified Tetrad method.
        """
        return 1 / 6


class UnspecifiedTetrad(DiscriminationMethod):
    """Unspecified Tetrad method."""

    def __init__(self) -> None:
        """Initialize an unspecified tetrad discrimination method."""
        super().__init__(name="utetrad")

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the Unspecified Tetrad method.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            Probability of correct response.
        """
        delta = np.array([d])
        delta = delta.flatten()
        f1 = norm.pdf
        f2 = norm.cdf

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def _fi(z: np.ndarray) -> np.ndarray:
            return 2 * (f1(z) * (2 * f2(z) * f2(z - dr) - f2(z - dr) ** 2))

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapz(y, x)

        return 1 - i

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the Unspecified Tetrad method.
        """
        return 1 / 3


class DualPair(DiscriminationMethod):
    """Dual Pair method."""

    def __init__(self) -> None:
        """Initialize a dual pair discrimination method."""
        super().__init__(name="dualpair")

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the Dual Pair method.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            Probability of correct response.
        """
        return norm.cdf(d / 2) ** 2 + norm.cdf(-d / 2) ** 2

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the Dual Pair method.
        """
        return 1 / 2


class DuoTrio(DiscriminationMethod):
    """Duo-Trio method."""

    def __init__(self) -> None:
        """Initialize a Duo-Trio discrimination method."""
        super().__init__(name="duotrio")

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the Duo-Trio method.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            Probability of correct response.
        """
        x1 = d / np.sqrt(2)
        x2 = d / np.sqrt(6)
        return 1 - norm.cdf(x1) - norm.cdf(x2) + 2 * norm.cdf(x1) * norm.cdf(x2)

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the Duo-Trio method.
        """
        return 1 / 2


class MplusN(DiscriminationMethod):
    """M+N method."""

    def __init__(self, m: int, n: int, specified: bool = False):
        """Initialize a m+n discrimination method.

        Args:
            m: TODO.
            n: TODO.
            specified: TODO.
        """
        c = "S" if specified else "U"
        super().__init__(name=f"{m}+{n}({c})")

        self.m = m
        self.n = n
        self.specified = specified
        self.psy_func = mplusn.mplusn_mc(m, n, specified=specified)

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the M+N method.

        Args:
            d: The Thurstonian sensory distance.

        Returns:
            Probability of correct response.
        """
        return self.psy_func(d)

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the M+N method.
        """
        if self.m > self.n or self.specified:
            return 1 / scipy.special.binom(self.m + self.n, self.n)
        else:
            return 2 / scipy.special.binom(self.m + self.n, self.n)


METHOD: dict[str, type[DiscriminationMethod]] = {
    "triangle": Triangle,
    "two_afc": TwoAFC,
    "three_afc": ThreeAFC,
    "four_afc": FourAFC,
    "duotrio": DuoTrio,
    "dualpair": DualPair,
    "utetrad": UnspecifiedTetrad,
    "stetrad": SpecifiedTetrad,
    "m_afc": MAFC,
    "mplusn": MplusN,
}
