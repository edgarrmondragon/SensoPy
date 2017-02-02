import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from psychometric_func import TEST, PSYCH, GUESSING

index = np.arange(100)
columns = TEST

pc = 0.8

for test in TEST:
    print(test)
    p0 = GUESSING[test]
    print(" ".join([" " * 3] + ["{0: ^6}".format(j) for j in np.arange(0, 0.1, 0.01)]))
    for i in np.arange(0, 1, 0.1):
        b1, r = divmod(p0 * 10, 1)
        if i < b1 / 10:
            continue
        ds = ["{0:1.1f}".format(i)]
        for j in np.arange(0, 0.1, 0.01):
            pc = i + j
            if pc < p0:
                ds.append(" " * 6)
                continue
            # print(pc)
            dprime = fsolve(lambda d: PSYCH[test](d) - pc, 1.0)[0]
            ds.append("{0:1.4f}".format(dprime))
        print(" ".join(ds))
