"""Discrimination Test."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.misc import derivative
from scipy.optimize import fsolve
from scipy.stats import beta, binom

from .methods import DiscriminationMethod


class Statistic(NamedTuple):
    """Results of a discrimination test."""

    estimate: float
    stderr: float
    lower: float
    upper: float


class TestResults(NamedTuple):
    """Results of a discrimination test."""

    pg: float
    pc: Statistic
    pd: Statistic
    d_prime: Statistic
    p_value: float
    alpha: float
    power: float


class DiscriminationTest:
    """A one-tailed test for the difference in performance between two groups."""

    def __init__(self, method: DiscriminationMethod) -> None:
        # def __init__(self, correct, panelists, ):
        """Initialize a discrimination test.

        Args:
            method: The discrimination method to use.
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
        """Compute the confidence limits for the test statistics.

        Args:
            x: TODO.
            n: TODO.
            pc: TODO.
            pd: TODO.
            pg: TODO.
            d_prime: TODO.
            alpha: TODO.

        Returns:
            TODO.
        """
        pc_err = np.sqrt(pc * (1 - pc) / n)
        pd_err = pc_err / (1 - pg)
        der = derivative(self.method.psychometric_function, d_prime, dx=1e-6)
        d_prime_err = pc_err / der

        # Lower limits
        pc_lower = max(beta.ppf(alpha / 2, x, n - x + 1), pg)
        pd_lower = (pc_lower - pg) / (1 - pg)
        d_prime_lower = fsolve(
            lambda d: self.method.psychometric_function(d) - pc_lower, 1.0
        )[0]

        # Upper limits
        pc_upper = min(beta.ppf(1 - alpha / 2, x + 1, n - x), 1.0)
        pd_upper = (pc_upper - pg) / (1 - pg)
        d_prime_upper = fsolve(
            lambda d: self.method.psychometric_function(d) - pc_upper, 1.0
        )[0]

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
        """Perform the difference one-tailed test.

             pc <= pc0
        H0:  pd <= pd0
             d' <= d'0

             pc > pc0
        H1:  pd > pd0
             d' > d'0

        Args:
            x: TODO.
            n: TODO.
            pd0: TODO.
            conf_level: TODO.

        Returns:
            TODO.
        """
        alpha = 1 - conf_level

        pg = self.method.guessing
        pc = x / n
        pd = (pc - pg) / (1 - pg)
        d_prime = fsolve(lambda d: self.method.psychometric_function(d) - pc, 1.0)[0]

        def stats(x, n, pc, pg, alpha):
            pc0 = pg + (1 - pg) * pd0
            p_value = 1 - binom.cdf(x - 1, n, pc0)
            xcrit = binom.ppf(1 - alpha, n, pc0) + 1
            power = 1 - binom.cdf(xcrit - 1, n, pc)
            return p_value, power

        p_value, power = stats(x, n, pc, pg, alpha)
        pc_stats, pd_stats, d_prime_stats = self.limits(
            x,
            n,
            pc,
            pd,
            pg,
            d_prime,
            alpha,
        )

        return TestResults(
            pg,
            pc_stats,
            pd_stats,
            d_prime_stats,
            p_value,
            alpha,
            power,
        )

    def equivalence(
        self,
        x: int,
        n: int,
        pd0: float = 0,
        conf_level: float = 0.95,
    ) -> TestResults:
        """Perform the equivalence one-tailed test.

             pc >= pc0
        H0:  pd >= pd0
             d' >= d'0

             pc < pc0
        H1:  pd < pd0
             d' < d'0

        Args:
            x: TODO.
            n: TODO.
            pd0: TODO.
            conf_level: TODO.

        Returns:
            TODO.
        """
        alpha = 1 - conf_level

        pg = self.method.guessing
        pc = x / n
        pd = (pc - pg) / (1 - pg)
        d_prime = fsolve(lambda d: self.method.psychometric_function(d) - pc, 1.0)[0]

        def stats(x, n, pc, pg, alpha):
            pc0 = pg + (1 - pg) * pd0
            p_value = binom.cdf(x, n, pc0)
            xcrit = binom.ppf(alpha, n, pc0) + 1
            power = binom.cdf(xcrit, n, pc)
            return p_value, power

        p_value, power = stats(x, n, pc, pg, alpha)
        pc_stats, pd_stats, d_prime_stats = self.limits(
            x,
            n,
            pc,
            pd,
            pg,
            d_prime,
            alpha,
        )

        return TestResults(
            pg,
            pc_stats,
            pd_stats,
            d_prime_stats,
            p_value,
            alpha,
            power,
        )
