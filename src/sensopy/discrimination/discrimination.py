"""Discrimination Test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.stats import beta, binom

if TYPE_CHECKING:
    from .methods import DiscriminationMethod


@dataclass(slots=True)
class Statistic:
    """Point estimate and confidence interval for a single test statistic."""

    estimate: float
    stderr: float
    lower: float
    upper: float


@dataclass(slots=True)
class TestResults:
    """Full set of estimates from a discrimination test."""

    pg: float
    pc: Statistic
    pd: Statistic
    d_prime: Statistic
    p_value: float
    alpha: float
    power: float


@dataclass(slots=True)
class SimulationResults:
    """Outcome of simulating many discrimination tests."""

    x: npt.NDArray[np.int_]
    p_values: npt.NDArray[np.float64]
    alpha: float
    power: float


class DiscriminationTest:
    """Difference and equivalence tests for a single sensory discrimination method.

    Implements the one-tailed difference test (H1: P_c > P_c0) and the one-tailed
    equivalence test (H1: P_c < P_c0) described in Bi (2015, Chapters 4-5).

    The test produces estimates of P_c, p_d, and d' together with their standard
    errors and confidence limits derived from the binomial model and the delta method
    (Bi, 2015, §2.3).
    """

    def _solve_d_prime(self, pc: float, d_max: float = 10.0) -> float:
        """Invert the psychometric function for a given proportion correct.

        The psychometric function is monotonically increasing from the guessing
        probability at d' = 0 towards 1, so the root is bracketed and found with
        Brent's method. Proportions at or beyond the boundaries have no finite
        interior root and map to 0 and infinity respectively.

        Returns:
            The Thurstonian distance d-prime for the given P_c.
        """
        f = self.method.psychometric_function
        if pc <= self.method.guessing:
            return 0.0
        if pc >= 1.0 or f(d_max) <= pc:
            return float("inf")
        return brentq(lambda d: f(d) - pc, 0.0, d_max)

    def __init__(self, method: DiscriminationMethod) -> None:
        """Initialize a discrimination test.

        Args:
            method: The sensory discrimination method to use.
        """
        self.method = method

    def limits(
        self,
        x: int,
        n: int,
        pc: float,
        pd: float,
        pg: float,
        d_prime: float,
        alpha: float,
    ) -> tuple[Statistic, Statistic, Statistic]:
        """Compute point estimates, standard errors, and confidence limits.

        Standard errors for P_c and p_d use the binomial formula. The standard
        error for d' is obtained via the delta method (Bi, 2015, §2.3):

            SE(d') = SE(P_c) / f'(d')

        where f'(d') is the derivative of the psychometric function, approximated
        here by a central finite difference. Confidence limits for P_c use the
        exact beta distribution (Clopper-Pearson) interval.

        Args:
            x: Number of correct responses.
            n: Number of panelists (trials).
            pc: Observed proportion correct P_c = x / n.
            pd: Observed proportion of discriminators p_d.
            pg: Guessing probability P_g for the method.
            d_prime: Estimated Thurstonian distance d'.
            alpha: Significance level (1 - confidence level).

        Returns:
            A tuple of (pc_stat, pd_stat, d_prime_stat), each a Statistic
            namedtuple with fields (estimate, stderr, lower, upper).
        """
        pc_err = np.sqrt(pc * (1 - pc) / n)
        pd_err = pc_err / (1 - pg)
        # The delta method breaks down at the boundaries, where d' is 0 (flat
        # psychometric function) or infinite.
        d_prime_err = float("inf")
        if 0 < d_prime < float("inf"):
            dx = 1e-6
            f = self.method.psychometric_function
            der = (f(d_prime + dx) - f(d_prime - dx)) / (2 * dx)
            d_prime_err = pc_err / der

        # Lower limits
        pc_lower = max(beta.ppf(alpha / 2, x, n - x + 1) if x > 0 else 0.0, pg)
        pd_lower = (pc_lower - pg) / (1 - pg)
        d_prime_lower = self._solve_d_prime(pc_lower)

        # Upper limits
        pc_upper = min(beta.ppf(1 - alpha / 2, x + 1, n - x) if x < n else 1.0, 1.0)
        pd_upper = (pc_upper - pg) / (1 - pg)
        d_prime_upper = self._solve_d_prime(pc_upper)

        return (
            Statistic(pc, pc_err, pc_lower, pc_upper),
            Statistic(pd, pd_err, pd_lower, pd_upper),
            Statistic(d_prime, d_prime_err, d_prime_lower, d_prime_upper),
        )

    def difference(
        self,
        x: int,
        n: int,
        pd0: float = 0,
        conf_level: float = 0.95,
    ) -> TestResults:
        """One-tailed difference test (Bi, 2015, Chapter 4).

        Tests whether the sensory difference exceeds a specified threshold pd0:

            H0: p_d ≤ pd0  (i.e., P_c ≤ P_c0)
            H1: p_d > pd0  (i.e., P_c > P_c0)

        where P_c0 = P_g + (1 - P_g) · pd0.

        Args:
            x: Number of correct responses.
            n: Number of panelists (trials).
            pd0: Null-hypothesis proportion of discriminators (default 0).
            conf_level: Confidence level for the interval estimates (default 0.95).

        Returns:
            TestResults with pg, pc, pd, d_prime (each a Statistic), p_value,
            alpha, and power.
        """
        alpha = 1 - conf_level

        pg = self.method.guessing
        pc = x / n
        pd = (pc - pg) / (1 - pg)
        d_prime = self._solve_d_prime(pc)

        def stats(x: int, n: int, pc: float, pg: float, alpha: float) -> tuple[float, float]:
            pc0 = pg + (1 - pg) * pd0
            p_value = 1 - binom.cdf(x - 1, n, pc0)
            xcrit = binom.ppf(1 - alpha, n, pc0) + 1
            power = 1 - binom.cdf(xcrit - 1, n, pc)
            return p_value, power

        p_value, power = stats(x, n, pc, pg, alpha)
        pc_stats, pd_stats, d_prime_stats = self.limits(x, n, pc, pd, pg, d_prime, alpha)

        return TestResults(pg, pc_stats, pd_stats, d_prime_stats, p_value, alpha, power)

    def equivalence(
        self,
        x: int,
        n: int,
        pd0: float = 0,
        conf_level: float = 0.95,
    ) -> TestResults:
        """One-tailed equivalence (similarity) test (Bi, 2015, Chapter 5).

        Tests whether the sensory difference is smaller than a specified threshold pd0:

            H0: p_d ≥ pd0  (i.e., P_c ≥ P_c0)
            H1: p_d < pd0  (i.e., P_c < P_c0)

        where P_c0 = P_g + (1 - P_g) · pd0.

        Args:
            x: Number of correct responses.
            n: Number of panelists (trials).
            pd0: Null-hypothesis proportion of discriminators (default 0).
            conf_level: Confidence level for the interval estimates (default 0.95).

        Returns:
            TestResults with pg, pc, pd, d_prime (each a Statistic), p_value,
            alpha, and power.
        """
        alpha = 1 - conf_level

        pg = self.method.guessing
        pc = x / n
        pd = (pc - pg) / (1 - pg)
        d_prime = self._solve_d_prime(pc)

        def stats(x: int, n: int, pc: float, pg: float, alpha: float) -> tuple[float, float]:
            pc0 = pg + (1 - pg) * pd0
            p_value = binom.cdf(x, n, pc0)
            # Largest count whose lower-tail p-value is still <= alpha
            xcrit = binom.ppf(alpha, n, pc0)
            if binom.cdf(xcrit, n, pc0) > alpha:
                xcrit -= 1
            power = binom.cdf(xcrit, n, pc)
            return p_value, power

        p_value, power = stats(x, n, pc, pg, alpha)
        pc_stats, pd_stats, d_prime_stats = self.limits(x, n, pc, pd, pg, d_prime, alpha)

        return TestResults(pg, pc_stats, pd_stats, d_prime_stats, p_value, alpha, power)

    def simulate(
        self,
        d: float,
        n: int,
        size: int = 1000,
        *,
        kind: Literal["difference", "equivalence"] = "difference",
        pd0: float = 0,
        conf_level: float = 0.95,
        rng: np.random.Generator | int | None = None,
    ) -> SimulationResults:
        """Simulate repeated discrimination tests and estimate the empirical power.

        Draws ``size`` experiments of ``n`` panelists with true distance ``d``, computes
        the one-tailed exact binomial p-value of each against ``pd0``, and reports the
        fraction that reject the null hypothesis at ``alpha = 1 - conf_level``.

        Args:
            d: True Thurstonian distance d' used to generate the data.
            n: Number of panelists per simulated test.
            size: Number of tests to simulate (default 1000).
            kind: ``"difference"`` (H1: P_c > P_c0) or ``"equivalence"`` (H1: P_c < P_c0).
            pd0: Null-hypothesis proportion of discriminators (default 0).
            conf_level: Confidence level (default 0.95).
            rng: A NumPy random generator, or a seed used to create one.

        Returns:
            SimulationResults with the simulated counts, p-values, alpha and power.

        Raises:
            ValueError: If ``kind`` is not recognized or ``size`` is not positive.
        """
        if kind not in ("difference", "equivalence"):
            msg = "kind must be 'difference' or 'equivalence'"
            raise ValueError(msg)
        if size < 1:
            msg = "size must be a positive integer"
            raise ValueError(msg)

        alpha = 1 - conf_level
        pg = self.method.guessing
        pc0 = pg + (1 - pg) * pd0
        x = self.method.simulate(d, n, size=size, rng=rng)
        p_values = 1 - binom.cdf(x - 1, n, pc0) if kind == "difference" else binom.cdf(x, n, pc0)
        return SimulationResults(x, p_values, alpha, float(np.mean(p_values <= alpha)))
