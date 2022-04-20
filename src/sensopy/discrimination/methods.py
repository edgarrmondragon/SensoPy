"""Discrimination methods."""

from __future__ import annotations

from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np
import scipy.special
from scipy.integrate import trapezoid
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
    """Triangle method.

    The Triangular (Triangle) method (Dawson and Harris 1951, Peryam 1958):

    Three samples of two products, A and B, are presented to each panelist. Two of them
    are the same. The possible sets of samples are AAB, ABA, BAA, ABB, BAB, and BBA.
    The panelist is asked to select the odd sample. The panelist is required to select
    one sample even if he or she cannot identify the odd one.
    """

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
        f1 = norm.pdf
        f2 = norm.cdf

        def _fi(z: np.ndarray) -> np.ndarray:
            x1 = -z * np.sqrt(3) + np.sqrt(2 / 3)
            x2 = -z * np.sqrt(3) - np.sqrt(2 / 3)
            return 2 * (f2(x1 * d) + f2(x2 * d)) * f1(z)

        x = np.linspace(0, 200, 10000)
        y = _fi(x)
        i = trapezoid(y, x)

        return i

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the Triangle method.
        """
        return 1 / 3


class TwoAFC(DiscriminationMethod):
    """Two-Alternative Forced-Choice method.

    The Two-Alternative Forced Choice (2-AFC) method (Green and Swets 1966):

    This method is also called the paired comparison method (Dawson and Harris 1951,
    Peryam 1958). With this method, the panelist receives a pair of coded samples,
    A and B, for comparison on the basis of some specified sensory characteristic.
    The possible pairs are AB and BA. The panelist is asked to select the sample with
    the strongest (or weakest) sensory characteristic. The panelist is required to
    select one even if he or she cannot detect the difference.
    """

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
    """Three-Alternative Forced-Choice method.

    The Three-Alternative Forced Choice (3-AFC) method (Green and Swets 1966):

    Three samples of two products, A and B, are presented to each panelist. Two of them
    are the same. The possible sets of samples are AAB, ABA, BAA or ABB, BAB, BBA.
    The panelist is asked to select the sample with the strongest or the weakest
    characteristic. The panelist has to select a sample even if he or she cannot
    identify the one with the strongest or the weakest sensory characteristic.
    """

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

        def _fi(z: np.ndarray):
            return (norm.cdf(z) ** 2) * norm.pdf(z - d)

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapezoid(y, x)

        return i

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the 3-AFC method.
        """
        return 1 / 3


class FourAFC(DiscriminationMethod):
    """Four-Alternative Forced-Choice method.

    The Four-Alternative Forced Choice (4-AFC) method (Swets 1959):

    Four samples of two products, A and B, are presented to each panelist. Three of
    them are the same. The possible sets of samples are AAAB, AABA, ABAA, BAAA or BBBA,
    BBAB, BABB, ABBB. The panelist is asked to select the sample with the strongest or
    the weakest characteristic. The panelist is required to select a sample even if he
    or she cannot identify the one with the strongest or weakest sensory characteristic.
    """

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

        def _fi(z: np.ndarray) -> np.ndarray:
            return (norm.cdf(z) ** 3) * norm.pdf(z - d)

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapezoid(y, x)

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

        def _fi(z: np.ndarray) -> np.ndarray:
            return (norm.cdf(z) ** (self.m - 1)) * norm.pdf(z - d)

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapezoid(y, x)

        return i

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the m-AFC method.
        """
        return 1 / self.m


class SpecifiedTetrad(DiscriminationMethod):
    """Specified Tetrad method.

    The Specified Tetrad method (Wood 1949):

    Four stimuli, two of A and two of B, are used, where A and B are confusable and vary
    in the relative strengths of their sensory attributes. Panelists are told that there
    are two pairs of putatively identical stimuli and to indicate the two stimuli of
    specified A or B.
    """

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
        f1 = norm.pdf
        f2 = norm.cdf

        def _fi(z: np.ndarray) -> np.ndarray:
            return 2 * (f1(z) * f2(z) * (2 * f2(z - d) - f2(z - d) ** 2))

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapezoid(y, x)

        return 1 - i

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the Specified Tetrad method.
        """
        return 1 / 6


class UnspecifiedTetrad(DiscriminationMethod):
    """Unspecified Tetrad method.

    The Unspecified Tetrad method (Lockhart 1951):

    Four stimuli, two of A and two of B, are used, where A and B are confusable and vary
    in the relative strengths of their sensory attributes. Panelists are told that there
    are two pairs of putatively identical stimuli and to sort them into their pairs.
    """

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
        f1 = norm.pdf
        f2 = norm.cdf

        def _fi(z: np.ndarray) -> np.ndarray:
            return 2 * (f1(z) * (2 * f2(z) * f2(z - d) - f2(z - d) ** 2))

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapezoid(y, x)

        return 1 - i

    @property
    def guessing(self) -> float:
        """Get guessing rate.

        Returns:
            Guessing rate for the Unspecified Tetrad method.
        """
        return 1 / 3


class DualPair(DiscriminationMethod):
    """Dual Pair method.

    The Dual Pair (4IAX) method (Macmillan et al. 1977):

    Two pairs of samples are presented simultaneously to the panelist. One pair is
    composed of samples of the same stimuli, AA or BB, while the other is composed of
    samples of differ- ent stimuli, AB or BA. The panelist is told to select the most
    different pair of the two pairs.
    """

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
    """Duo-Trio method.

    The Duo-Trio method (Dawson and Harris 1951, Peryam 1958):

    Three samples of two products, A and B, are presented to each panelist. Two of
    them are the same. The possible sets of samples are A: AB, A: BA, B: AB, and B: BA.
    The first one is labeled as the “control.” The panelist is asked which of the two
    test samples is the same as the control sample. The panelist is required to select
    one sample to match the “control” sample even if he or she cannot identify which is
    the same as the control.
    """

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
    """M+N method.

    The “M + N” method (Lockhart 1951):

    M + N samples with M sample A and N sample B are presented. The panelist is told to
    divide the samples into two groups, of A and B. There are two versions of the
    method: specified and unspecified.

    This is a generalization of many forced-choice discrimination methods, including
    the Multiple-Alternative Forced Choice (m-AFC), Triangle, and Specified and
    Unspecified Tetrad.

    The “M + N” with larger M and N can be regarded as a specific discrimination method
    with a new model. Unlike the conventional difference tests using the “M + N” with
    small M and N based on a binomial model, the “M + N” with larger M and N (M = N > 3)
    can reach a statistical significance in a single trial for only one “M + N” sample
    set based on a hypergeometric model.
    """

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
