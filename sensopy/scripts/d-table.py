import numpy as np
import pandas as pd
from scipy.optimize import fsolve

from psychometric_func import TEST, PSYCH, GUESSING

# d = np.linspace(0, 5, 100)
d = np.arange(0, 8, 0.05)
print(d)
df = pd.DataFrame()
df["d'"] = pd.Series(d)

for test in TEST:
    df[test] = PSYCH[test](d)
    
df.to_csv("Thurstone-d'.csv")
