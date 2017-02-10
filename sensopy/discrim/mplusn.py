import numpy as np
from scipy import interpolate

RAND_SEED = 12345
SAMPLE_SIZE = 100000

#------------------------------------------------------------------------------
# "M plus N" simulation
#------------------------------------------------------------------------------

def mplusn_mc(m, n, specified=False, max_delta=5, nsteps=300,
              seed=RAND_SEED, sample_size=SAMPLE_SIZE):
    """Monte Carlo simulation for M + N method"""
    if specified:
        c = "S"
    else:
        c = "U"
    if m < n:
        raise ValueError("Invalid combination of parameters. M >= N expected.")
    
    delta = np.linspace(0, max_delta, nsteps)
    prop = []
    k = m - n
    
    # Seed the random number generator
    np.random.seed(seed=seed)
    for d in delta:
        # Samples from A ~ N(0,1)
        A = np.random.randn(n, SAMPLE_SIZE)
        
        # Samples from B ~ N(d,1)
        B = np.random.randn(m, SAMPLE_SIZE) + d
        
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

