import numpy as np
from scipy.optimize import fsolve
from scipy.misc import derivative
from scipy.stats import binom
import sys
import csv

from psychometric_func import TEST, PSYCH, GUESSING

index = np.arange(100)
columns = TEST

# alpha = 0.05
power = 0.80

def get_power(pg, pc, n, alpha=0.05, test="difference", pd0=0):
    
    pc0 = pg + (1 - pg) * pd0
    
    if test == "difference":
        xcrit = binom.ppf(1 - alpha, n, pc0) + 1
        bet = binom.cdf(xcrit - 1, n, pc)
    
    elif test == "similarity":
        xcrit = binom.ppf(alpha, n, pc0) - 1
        bet = 1 - binom.cdf(xcrit, n, pc)
    
    return xcrit, 1 - bet
    
def min_sample(pg, pc, alpha=0.05, power=0.8, test="difference", pd0=0):
    sample_size = 1
    while True:
        xcrit, statpow = get_power(pg, pc, sample_size, alpha=alpha, test=test, pd0=pd0)
        sample_size += 1
        # print(sample_size)
        if sample_size == 1000:
            return ">1000"
        if statpow >= power:
            return sample_size

            

def main():
    print(get_power(1 / 3, 19 / 30, 30))
    
    b = min_sample(1 / 3, 19 / 30)
    print(b)
    
    b = min_sample(1 / 3, 6 / 30, test="similarity", pd0=0.05)
    print(b)
    
    data = []
    
    # with open(sys.argv[1], "w", newline="") as csvfile:
        # writer = csv.writer(csvfile)
        # writer.writerow(["d"] + TEST)
        # for delta in np.arange(0.3, 2.05, 0.05):
            # print(delta)
            # row = [delta]
            # for test in TEST:
                # pc = PSYCH[test](delta)
                # pg = GUESSING[test]
                # print(test, delta, pc, pg)
                # min_n = min_sample(pg, pc, test="similarity", pd0=0.05)
                # row.append(min_n)
            # writer.writerow(row)
                
    
if __name__ == "__main__":
    main()
