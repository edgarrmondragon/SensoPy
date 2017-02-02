import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

rand_seed = 12345
N = 100000

m = 3
n = 2
k = m - n

#------------------------------------------------------------------------------

# "2 out of 5" simulation
#------------------------------------------------------------------------------
print('*' * 80)
# np.random.seed(seed=rand_seed)

delta = np.linspace(0, 5, 200)
prop = []
sigma = 1
p0 = 0.5

# a = np.array([[12,41,64],[63,2,52],[23,145,12]])
# b = np.array([[12,41,64],[63,2,52],[23,145,12],[25,4,19]])

# print(np.amin(a, axis=0))
# print(b)
# print(np.sort(b, axis=0))
# print(b)

np.random.seed(seed=12345)

specified = True

for d in delta:
    A = np.random.randn(n, N)
    B = sigma * np.random.randn(m, N) + d
    
    A.sort(axis=0)
    B.sort(axis=0)
    
    if specified:
        pc = np.mean(A[-1] < B[0])
    elif k == 0:
        pc = np.mean((A[-1] < B[0]) | A[0] > B[-1])
    elif k > 0:
        cond1 = ((B[k] - B[k - 1]) < (B[0] - A[n - 1])) & (A[-1] < B[0])
        cond2 = ((B[n] - B[n - 1]) < (A[0] - B[m - 1])) & (A[0] > B[-1])
        pc = np.mean(cond1 | cond2)
        
    prop.append(pc)
    
    # print(d, pc)

f = interpolate.interp1d(delta, prop)
print(f(1.5270))

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
