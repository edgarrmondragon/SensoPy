"""M plus N generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import interpolate

if TYPE_CHECKING:
    from collections.abc import Callable

SAMPLE_SIZE = 100000

# ------------------------------------------------------------------------------
# "M plus N" simulation
# ------------------------------------------------------------------------------


def mplusn_mc(
    m: int,
    n: int,
    specified: bool = False,
    max_delta: float = 10,
    steps: int = 300,
    seed: int | None = None,
    sample_size: int = SAMPLE_SIZE,
) -> Callable[[float], float]:
    """Monte Carlo simulation for M + N method.

    Args:
        m: TODO.
        n: TODO.
        specified: TODO.
        max_delta: TODO.
        steps: TODO.
        seed: TODO.
        sample_size: TODO.

    Returns:
        TODO.

    Raises:
        ValueError: If `m` is smaller than `n`.
    """
    if m < n:
        raise ValueError("Invalid combination of parameters. M >= N expected.")

    delta = np.linspace(0, max_delta, steps)
    prop = []
    k = m - n

    def _func1(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
        return np.mean(a[-1] < b[0])  # type: ignore[no-any-return]

    def _func2(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
        cond1 = a[-1] < b[0]
        cond2 = a[0] > b[-1]
        return np.mean(cond1 | cond2)  # type: ignore[no-any-return]

    def _func3(a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
        cond1 = ((b[k] - b[k - 1]) < (b[0] - a[n - 1])) & (a[-1] < b[0])
        cond2 = ((b[n] - b[n - 1]) < (a[0] - b[m - 1])) & (a[0] > b[-1])
        return np.mean(cond1 | cond2)  # type: ignore[no-any-return]

    # Specified test
    if specified:
        p = _func1
    # Test with M = N
    elif k == 0:
        p = _func2
    # Test with M > N
    else:
        p = _func3

    # Seed the random number generator
    rng = np.random.default_rng(seed=seed)
    for d in delta:
        # Samples from A ~ N(0,1)
        a = rng.standard_normal(size=(n, sample_size))

        # Samples from B ~ N(d,1)
        b = rng.standard_normal(size=(m, sample_size)) + d

        # Sort corresponding n-tuples
        a.sort(axis=0)
        b.sort(axis=0)
        pc = p(a, b)

        prop.append(pc)

    return interpolate.interp1d(delta, prop)  # type: ignore[return-value]
