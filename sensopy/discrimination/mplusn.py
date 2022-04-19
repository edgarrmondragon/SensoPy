"""M plus N generation."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import interpolate

RAND_SEED = 12345
SAMPLE_SIZE = 100000

# ------------------------------------------------------------------------------
# "M plus N" simulation
# ------------------------------------------------------------------------------


def mplusn_mc(
    m: int,
    n: int,
    specified: bool = False,
    max_delta: float = 5,
    nsteps: int = 300,
    seed: int = RAND_SEED,
    sample_size: int = SAMPLE_SIZE,
) -> Callable:
    """Monte Carlo simulation for M + N method.

    Args:
        m: TODO.
        n: TODO.
        specified: TODO.
        max_delta: TODO.
        nsteps: TODO.
        seed: TODO.
        sample_size: TODO.

    Returns:
        TODO.

    Raises:
        ValueError: If `m` is smaller than `n`.
    """
    if m < n:
        raise ValueError("Invalid combination of parameters. M >= N expected.")

    delta = np.linspace(0, max_delta, nsteps)
    prop = []
    k = m - n

    # Seed the random number generator
    np.random.seed(seed=seed)
    for d in delta:
        # Samples from A ~ N(0,1)
        A = np.random.randn(n, sample_size)

        # Samples from B ~ N(d,1)
        B = np.random.randn(m, sample_size) + d

        # Sort corresponding n-tuples
        A.sort(axis=0)
        B.sort(axis=0)

        # Specified test
        if specified:
            pc = np.mean(A[-1] < B[0])
        # Test with M = N
        elif k == 0:
            cond1 = A[-1] < B[0]
            cond2 = A[0] > B[-1]
            pc = np.mean(cond1 | cond2)
        # Test with M > N
        elif k > 0:
            cond1 = ((B[k] - B[k - 1]) < (B[0] - A[n - 1])) & (A[-1] < B[0])
            cond2 = ((B[n] - B[n - 1]) < (A[0] - B[m - 1])) & (A[0] > B[-1])
            pc = np.mean(cond1 | cond2)

        prop.append(pc)

    f = interpolate.interp1d(delta, prop)
    return f
