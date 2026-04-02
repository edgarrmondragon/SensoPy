"""Discrimination methods."""

from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt
import scipy.special
from scipy.integrate import trapezoid
from scipy.stats import norm

from . import mplusn

__all__ = [
    "DUAL_PAIR",
    "DUO_TRIO",
    "FOUR_AFC",
    "SPECIFIED_TETRAD",
    "THREE_AFC",
    "TRIANGLE",
    "TWO_AFC",
    "UNSPECIFIED_TETRAD",
    "MPlusNMethod",
    "MultipleAFCMethod",
]


class DiscriminationMethod(abc.ABC):
    """A sensory discrimination method."""

    @abc.abstractmethod
    def psychometric_function(self, d: float) -> float:
        """Psychometric function relating d' to the probability of a correct response.

        Args:
            d: Thurstonian discriminal distance d' (δ), a method-independent measure
                of sensory difference/similarity (Bi, 2015, §2.1).

        Returns:
            Probability of a correct response P_c for the given d'.
        """
        ...

    @property
    @abc.abstractmethod
    def guessing(self) -> float:
        """Chance-level probability of a correct response when d' = 0."""
        ...

    def discriminators(self, d: float) -> float:
        """Proportion of discriminators (p_d) for a given d'.

        p_d is the proportion of the population that can reliably detect the
        sensory difference, estimated as (P_c - P_g) / (1 - P_g), where P_g
        is the guessing rate (Bi, 2015, §4.1).

        Args:
            d: Thurstonian discriminal distance d'.

        Returns:
            Proportion of discriminators p_d.
        """
        pc = self.psychometric_function(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)


class TriangleMethod(DiscriminationMethod):
    """Triangle (Triangular) discrimination method (Dawson and Harris 1951, Peryam 1958).

    Three samples of two products, A and B, are presented to each panelist. Two of them
    are the same. The possible sets of samples are AAB, ABA, BAA, ABB, BAB, and BBA.
    The panelist is asked to select the odd sample and must select one even if they
    cannot identify it (Bi, 2015, §1.5.1d).

    The psychometric function is (Bi, 2015, eq. 2.2.5):

        P_c = 2 ∫_0^∞ φ(x) { Φ[-√3·x + √(2/3)·δ] + Φ[-√3·x - √(2/3)·δ] } dx

    Guessing probability: 1/3.
    """

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the Triangle method (Bi, 2015, eq. 2.2.5).

        Args:
            d: Thurstonian discriminal distance d'.

        Returns:
            Probability of a correct response P_c.
        """
        f1 = norm.pdf
        f2 = norm.cdf

        def _fi(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            x1 = -z * np.sqrt(3) + np.sqrt(2 / 3) * d
            x2 = -z * np.sqrt(3) - np.sqrt(2 / 3) * d
            return 2 * (f2(x1) + f2(x2)) * f1(z)  # type: ignore[no-any-return]

        x = np.linspace(0, 200, 10000)
        y = _fi(x)
        return trapezoid(y, x)  # type: ignore[no-any-return]  # type: ignore[no-any-return]

    @property
    def guessing(self) -> float:
        """Chance-level probability for the Triangle method (1/3)."""
        return 1 / 3


class TwoAFCMethod(DiscriminationMethod):
    """Two-Alternative Forced Choice (2-AFC) method (Green and Swets 1966).

    Also called the paired comparison method (Dawson and Harris 1951, Peryam 1958).
    The panelist receives a pair of coded samples, A and B, for comparison on the basis
    of a specified sensory characteristic. The possible pairs are AB and BA. The panelist
    selects the sample with the strongest (or weakest) characteristic and must choose
    even if they cannot detect a difference (Bi, 2015, §1.5.1a).

    The psychometric function is (Bi, 2015, eq. 2.2.1):

        P_c = Φ(δ / √2)

    Guessing probability: 1/2.
    """

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the 2-AFC method (Bi, 2015, eq. 2.2.1).

        Args:
            d: Thurstonian discriminal distance d'.

        Returns:
            Probability of a correct response P_c.
        """
        return norm.cdf(d / np.sqrt(2))  # type: ignore[no-any-return]

    @property
    def guessing(self) -> float:
        """Chance-level probability for the 2-AFC method (1/2)."""
        return 1 / 2


class ThreeAFCMethod(DiscriminationMethod):
    """Three-Alternative Forced Choice (3-AFC) method (Green and Swets 1966).

    Three samples of two products, A and B, are presented to each panelist. Two of them
    are the same. The possible sets of samples are AAB, ABA, BAA or ABB, BAB, BBA.
    The panelist selects the sample with the strongest or weakest characteristic and must
    choose even if they cannot identify it (Bi, 2015, §1.5.1b).

    The psychometric function is (Bi, 2015, eq. 2.2.2):

        P_c = ∫_{-∞}^{∞} Φ²(u) φ(u - δ) du

    Guessing probability: 1/3.
    """

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the 3-AFC method (Bi, 2015, eq. 2.2.2).

        Args:
            d: Thurstonian discriminal distance d'.

        Returns:
            Probability of a correct response P_c.
        """

        def _fi(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return (norm.cdf(z) ** 2) * norm.pdf(z - d)

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        return trapezoid(y, x)  # type: ignore[no-any-return]

    @property
    def guessing(self) -> float:
        """Chance-level probability for the 3-AFC method (1/3)."""
        return 1 / 3


class FourAFCMethod(DiscriminationMethod):
    """Four-Alternative Forced Choice (4-AFC) method (Swets 1959).

    Four samples of two products, A and B, are presented to each panelist. Three of them
    are the same. The possible sets of samples are AAAB, AABA, ABAA, BAAA or BBBA,
    BBAB, BABB, ABBB. The panelist selects the sample with the strongest or weakest
    characteristic and must choose even if they cannot identify it (Bi, 2015, §1.5.1c).

    The psychometric function is (Bi, 2015, eq. 2.2.3):

        P_c = ∫_{-∞}^{∞} Φ³(u) φ(u - δ) du

    Guessing probability: 1/4.
    """

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the 4-AFC method (Bi, 2015, eq. 2.2.3).

        Args:
            d: Thurstonian discriminal distance d'.

        Returns:
            Probability of a correct response P_c.
        """

        def _fi(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return (norm.cdf(z) ** 3) * norm.pdf(z - d)

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        return trapezoid(y, x)  # type: ignore[no-any-return]

    @property
    def guessing(self) -> float:
        """Chance-level probability for the 4-AFC method (1/4)."""
        return 1 / 4


class MultipleAFCMethod(DiscriminationMethod):
    """m-Alternative Forced Choice (m-AFC) method.

    Generalisation of the AFC family. m samples are presented; (m - 1) are from
    product A and one from product B. The panelist selects the sample with the
    strongest (or weakest) characteristic (Bi, 2015, §1.5.1).

    The psychometric function is (Bi, 2015, eq. 2.2.3 generalised):

        P_c = ∫_{-∞}^{∞} Φ^(m-1)(u) φ(u - δ) du

    Guessing probability: 1/m.
    """

    def __init__(self, m: int) -> None:
        """Initialize an m-AFC discrimination method.

        Args:
            m: Number of alternatives (must be ≥ 2).
        """
        self.m = m

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the m-AFC method (Bi, 2015, eq. 2.2.3 generalised).

        Args:
            d: Thurstonian discriminal distance d'.

        Returns:
            Probability of a correct response P_c.
        """

        def _fi(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return (norm.cdf(z) ** (self.m - 1)) * norm.pdf(z - d)

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        return trapezoid(y, x)  # type: ignore[no-any-return]

    @property
    def guessing(self) -> float:
        """Chance-level probability for the m-AFC method (1/m)."""
        return 1 / self.m


class SpecifiedTetradMethod(DiscriminationMethod):
    """Specified Tetrad method (Wood 1949).

    Four stimuli — two of A and two of B — are presented. A and B are confusable and
    vary in the relative strengths of their sensory attributes. Panelists are told that
    there are two pairs of putatively identical stimuli and must indicate the two stimuli
    of the specified product (A or B) (Bi, 2015, §1.5.1g).

    The psychometric function is (Bi, 2015, eq. 2.2.11):

        P_c = 1 - 2 ∫_{-∞}^{∞} φ(x) Φ(x) { 2Φ(x - δ) - [Φ(x - δ)]² } dx

    Guessing probability: 1/6.
    """

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the Specified Tetrad method (Bi, 2015, eq. 2.2.11).

        Args:
            d: Thurstonian discriminal distance d'.

        Returns:
            Probability of a correct response P_c.
        """
        f1 = norm.pdf
        f2 = norm.cdf

        def _fi(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return 2 * (f1(z) * f2(z) * (2 * f2(z - d) - f2(z - d) ** 2))

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapezoid(y, x)

        return 1 - i  # type: ignore[no-any-return]

    @property
    def guessing(self) -> float:
        """Chance-level probability for the Specified Tetrad method (1/6)."""
        return 1 / 6


class UnspecifiedTetrad(DiscriminationMethod):
    """Unspecified Tetrad method (Lockhart 1951).

    Four stimuli — two of A and two of B — are presented. A and B are confusable and
    vary in the relative strengths of their sensory attributes. Panelists are told that
    there are two pairs of putatively identical stimuli and must sort them into their
    pairs, without being told which product to identify (Bi, 2015, §1.5.1f).

    The psychometric function is (Bi, 2015, eq. 2.2.8):

        P_c = 1 - 2 ∫_{-∞}^{∞} φ(x) { 2Φ(x)Φ(x - δ) - [Φ(x - δ)]² } dx

    Guessing probability: 1/3.
    """

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the Unspecified Tetrad method (Bi, 2015, eq. 2.2.8).

        Args:
            d: Thurstonian discriminal distance d'.

        Returns:
            Probability of a correct response P_c.
        """
        f1 = norm.pdf
        f2 = norm.cdf

        def _fi(z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return 2 * (f1(z) * (2 * f2(z) * f2(z - d) - f2(z - d) ** 2))

        x = np.linspace(-100, 100, 10000)
        y = _fi(x)
        i = trapezoid(y, x)

        return 1 - i  # type: ignore[no-any-return]

    @property
    def guessing(self) -> float:
        """Chance-level probability for the Unspecified Tetrad method (1/3)."""
        return 1 / 3


class DualPairMethod(DiscriminationMethod):
    """Dual Pair (4IAX) method (Macmillan et al. 1977).

    Two pairs of samples are presented simultaneously to the panelist. One pair is
    composed of samples of the same stimulus (AA or BB); the other is composed of
    samples of different stimuli (AB or BA). The panelist selects the most different
    pair of the two (Bi, 2015, §1.5.1h).

    The psychometric function is (Bi, 2015, eq. 2.2.12):

        P_c = [Φ(δ/2)]² + [Φ(-δ/2)]²

    Guessing probability: 1/2.
    """

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the Dual Pair method (Bi, 2015, eq. 2.2.12).

        Args:
            d: Thurstonian discriminal distance d'.

        Returns:
            Probability of a correct response P_c.
        """
        return norm.cdf(d / 2) ** 2 + norm.cdf(-d / 2) ** 2

    @property
    def guessing(self) -> float:
        """Chance-level probability for the Dual Pair method (1/2)."""
        return 1 / 2


class DuoTrioMethod(DiscriminationMethod):
    """Duo-Trio method (Dawson and Harris 1951, Peryam 1958).

    Three samples of two products, A and B, are presented to each panelist. Two of them
    are the same. The possible sets of samples are A:AB, A:BA, B:AB, and B:BA. The first
    sample is labelled the "control." The panelist selects which of the two test samples
    matches the control and must choose even if they cannot identify the match
    (Bi, 2015, §1.5.1e).

    The psychometric function is (Bi, 2015, eq. 2.2.4):

        P_c = 1 - Φ(δ/√2) - Φ(δ/√6) + 2 Φ(δ/√2) Φ(δ/√6)

    Guessing probability: 1/2.
    """

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the Duo-Trio method (Bi, 2015, eq. 2.2.4).

        Args:
            d: Thurstonian discriminal distance d'.

        Returns:
            Probability of a correct response P_c.
        """
        x1 = d / np.sqrt(2)
        x2 = d / np.sqrt(6)
        return 1 - norm.cdf(x1) - norm.cdf(x2) + 2 * norm.cdf(x1) * norm.cdf(x2)  # type: ignore[no-any-return]

    @property
    def guessing(self) -> float:
        """Chance-level probability for the Duo-Trio method (1/2)."""
        return 1 / 2


class MPlusNMethod(DiscriminationMethod):
    """M + N discrimination method (Lockhart 1951).

    M + N samples are presented: M samples of product A and N samples of product B.
    The panelist divides them into two groups of A and B. There are two versions:
    specified (panelist is told which group is A) and unspecified (Bi, 2015, §1.5.1i).

    This is a generalisation of many forced-choice methods, including m-AFC, Triangle,
    and both Tetrad variants. For small M and N a binomial model applies; for larger
    M and N (M = N > 3) a single set of samples can reach statistical significance
    under a hypergeometric model (Bi, 2015, §2.5).

    The psychometric function is estimated by Monte Carlo simulation (Bi, 2015, §2.5).

    Guessing probability: 1/C(M+N, N) for specified or M > N; 2/C(M+N, N) otherwise.
    """

    def __init__(self, m: int, n: int, specified: bool = False, *, seed: int | None = None) -> None:
        """Initialize an M + N discrimination method.

        Args:
            m: Number of samples from product A (must be ≥ n).
            n: Number of samples from product B.
            specified: If True, panelists are told which group is product A (specified
                version); otherwise the unspecified version is used.
            seed: Seed for the random number generator used in the Monte Carlo
                simulation of the psychometric function. Pass an integer for
                reproducible results; ``None`` (default) uses an unpredictable seed.
        """
        self.m = m
        self.n = n
        self.specified = specified
        self.psy_func = mplusn.mplusn_mc(m, n, specified=specified, seed=seed)

    def psychometric_function(self, d: float) -> float:
        """Psychometric function for the M + N method (Monte Carlo estimate).

        Args:
            d: Thurstonian discriminal distance d'.

        Returns:
            Probability of a correct response P_c (interpolated from simulation).
        """
        return self.psy_func(d)

    @property
    def guessing(self) -> float:
        """Chance-level probability for the M + N method.

        Returns 1/C(M+N, N) for the specified version or when M > N, and
        2/C(M+N, N) for the unspecified version when M = N (Bi, 2015, §2.5).
        """
        if self.m > self.n or self.specified:
            return float(1 / scipy.special.binom(self.m + self.n, self.n))
        return float(2 / scipy.special.binom(self.m + self.n, self.n))


TRIANGLE = TriangleMethod()
TWO_AFC = TwoAFCMethod()
THREE_AFC = ThreeAFCMethod()
FOUR_AFC = FourAFCMethod()
SPECIFIED_TETRAD = SpecifiedTetradMethod()
UNSPECIFIED_TETRAD = UnspecifiedTetrad()
DUAL_PAIR = DualPairMethod()
DUO_TRIO = DuoTrioMethod()
