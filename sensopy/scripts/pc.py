import numpy as np
from scipy.optimize import fsolve
from scipy.misc import derivative
import csv

from psychometric_func import TEST, PSYCH, GUESSING

index = np.arange(100)
columns = TEST

pc = 0.8

def make_table(test):
    table = []
    p0 = GUESSING[test]
    table.append([""] + [j for j in np.arange(0, 0.1, 0.01)])
    
    for i in np.arange(0, 1, 0.1):
        b1, r = divmod(p0 * 10, 1)
        if i < b1 / 10:
            continue
        
        ds = [i]
        Bs = [""]
        
        for j in np.arange(0, 0.1, 0.01):
            pc = i + j
            if pc < p0:
                ds.append("")
                Bs.append("")
                continue
            
            dprime = fsolve(lambda d: PSYCH[test](d) - pc, 1.0)[0]
            B = pc * (1 - pc) / (derivative(PSYCH[test], dprime, dx=1e-6) ** 2)
            ds.append(dprime)
            Bs.append(B)
        
        table.append(ds)
        table.append(Bs)
    
    return table
    

# for test in TEST:
    # print(test)
    # p0 = GUESSING[test]
    # print(" ".join([" " * 3] + ["{0: ^6}".format(j) for j in np.arange(0, 0.1, 0.01)]))
    # for i in np.arange(0, 1, 0.1):
        # b1, r = divmod(p0 * 10, 1)
        # if i < b1 / 10:
            # continue
        # ds = ["{0:1.1f}".format(i)]
        # Bs = ["{0:1.1f}".format(i)]
        # for j in np.arange(0, 0.1, 0.01):
            # pc = i + j
            # if pc < p0:
                # ds.append(" " * 7)
                # Bs.append(" " * 7)
                # continue
            # print(pc)
            # dprime = fsolve(lambda d: PSYCH[test](d) - pc, 1.0)[0]
            # B = pc * (1 - pc) / (derivative(PSYCH[test], dprime, dx=1e-6) ** 2)
            # ds.append("{0: >7.4f}".format(dprime))
            # Bs.append("{0: >7.4f}".format(B))
        # print(" ".join(ds))
        # print(" ".join(Bs))

def main():
    
    with open("d,B.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for test in TEST:
            writer.writerow([test])
            writer.writerow([])
            table = make_table(test)
            for row in table:
                writer.writerow(row)
            writer.writerow([])
                
    
if __name__ == "__main__":
    main()
