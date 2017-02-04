import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

RAND_SEED = 12345
N = 100000

m = 3
n = 2
k = m - n

#------------------------------------------------------------------------------

# "2 out of 5" simulation
#------------------------------------------------------------------------------
print('*' * 80)

delta = np.linspace(0, 5, 300)
prop = []
sigma = 1
p0 = 0.5

np.random.seed(seed=RAND_SEED)

specified = False

for d in delta:
    A = np.random.randn(n, N)
    B = sigma * np.random.randn(m, N) + d
    
    A.sort(axis=0)
    B.sort(axis=0)
    
    if specified:
        pc = np.mean(A[-1] < B[0])
    elif k == 0:
        cond1 = A[-1] < B[0]
        cond2 = A[0] > B[-1]
        pc = np.mean(cond1 | cond2)
    elif k > 0:
        cond1 = ((B[k] - B[k - 1]) < (B[0] - A[n - 1])) & (A[-1] < B[0])
        cond2 = ((B[n] - B[n - 1]) < (A[0] - B[m - 1])) & (A[0] > B[-1])
        pc = np.mean(cond1 | cond2)
        
    prop.append(pc)
    
    # print(d, pc)

f = interpolate.interp1d(delta, prop)
print(f(1.5270))
print(f(0))

# print(f(0.252))

plt.plot(delta, prop)
plt.show()




# A = np.random.randn(N)
# B = sigma * np.random.randn(N) + delta

# pc = np.mean(B > A)
# pd = (pc - p0) / (1 - p0)
# var_pc = np.sqrt((pc * (1.0 - pc) / N))

# print("pc = %f" % pc)
# print("pd = %f" % pd)
# print(abs(pc - norm.cdf(delta / np.sqrt(2))))  # Comparison against closed-form
