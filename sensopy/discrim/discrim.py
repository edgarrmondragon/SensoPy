import numpy as np
from scipy.stats import binom, beta
from scipy.optimize import fsolve
from scipy.misc import derivative
from collections import namedtuple
from . import methods


TestResults = namedtuple('TestResults', ["pg", "pc", "pd", "dprime", "pvalue", "alpha", "power"])
Statistic = namedtuple('Statistic', ["estimate", "stderr", "lower", "upper"])

class DiscriminationTest():
    """"""
    def __init__(self, method, **kwargs):
    # def __init__(self, correct, panelists, ):
        """"""
        self.method = methods.METHOD[method](**kwargs)
        
    def difference(self, x, n, pd0=0, conf_level=0.95):
        """      
        Difference one-tailed test
        
             pc <= pc0
        H0:  pd <= pd0
             d' <= d'0
             
             pc > pc0
        H1:  pd > pd0
             d' > d'0
             
        """
        alpha = 1 - conf_level
        
        pg = self.method.guessing
        pc = x / n
        pc0 = pg + (1 - pg) * pd0
        pd = (pc - pg) / (1 - pg)
        dprime = fsolve(lambda d: self.method.psychfunc(d) - pc, 1.0)[0]
        
        p_value = 1- binom.cdf(x - 1, n, pc0)
        xcrit = binom.ppf(1 - alpha, n, pc0) + 1
        power = 1- binom.cdf(xcrit - 1, n, pc)
        
        pc_err = np.sqrt(pc * (1 - pc) / n)
        pd_err = pc_err / (1 - pg)
        der = derivative(self.method.psychfunc, dprime, dx=1e-6)
        dprime_err = pc_err / der
        
        # Lower limits
        pc_lower = max(beta.ppf(alpha / 2, x, n - x + 1), pg)
        pd_lower = (pc_lower - pg) / (1 - pg)
        dprime_lower = fsolve(lambda d: self.method.psychfunc(d) - pc_lower, 1.0)[0]
        
        # Upper limits
        pc_upper = min(beta.ppf(1 - alpha / 2, x + 1, n - x), 1.0)
        pd_upper = (pc_upper - pg) / (1 - pg)
        dprime_upper = fsolve(lambda d: self.method.psychfunc(d) - pc_upper, 1.0)[0]
        
        results = TestResults(pg,
                              Statistic(pc, pc_err, pc_lower, pc_upper),
                              Statistic(pd, pd_err, pd_lower, pd_upper),
                              Statistic(dprime, dprime_err, dprime_lower, dprime_upper),
                              p_value,
                              alpha, power)
        return results
        
        
    def equivalence(self, x, n, pd0=0, conf_level=0.95):
        """      
        Equivalence one-tailed test
        
             pc >= pc0
        H0:  pd >= pd0
             d' >= d'0
             
             pc < pc0
        H1:  pd < pd0
             d' < d'0
             
        """
        alpha = 1 - conf_level
        
        pg = self.method.guessing
        pc = x / n
        pc0 = pg + (1 - pg) * pd0
        pd = (pc - pg) / (1 - pg)
        dprime = fsolve(lambda d: self.method.psychfunc(d) - pc, 1.0)[0]
        
        p_value = binom.cdf(x, n, pc0)
        xcrit = binom.ppf(alpha, n, pc0) + 1
        power = binom.cdf(xcrit, n, pc)
        
        pc_err = np.sqrt(pc * (1 - pc) / n)
        pd_err = pc_err / (1 - pg)
        der = derivative(self.method.psychfunc, dprime, dx=1e-6)
        dprime_err = pc_err / der
        
        # Lower limits
        pc_lower = max(beta.ppf(alpha / 2, x, n - x + 1), pg)
        pd_lower = (pc_lower - pg) / (1 - pg)
        dprime_lower = fsolve(lambda d: self.method.psychfunc(d) - pc_lower, 1.0)[0]
        
        # Upper limits
        pc_upper = min(beta.ppf(1 - alpha / 2, x + 1, n - x), 1.0)
        pd_upper = (pc_upper - pg) / (1 - pg)
        dprime_upper = fsolve(lambda d: self.method.psychfunc(d) - pc_upper, 1.0)[0]
        
        results = TestResults(pg,
                              Statistic(pc, pc_err, pc_lower, pc_upper),
                              Statistic(pd, pd_err, pd_lower, pd_upper),
                              Statistic(dprime, dprime_err, dprime_lower, dprime_upper),
                              p_value,
                              alpha, power)
        return results

            
def main():

    test = DiscriminationTest("triangle")
    t1 = test.difference(19, 30)
    print(t1)
    t2 = test.equivalence(19, 30)
    print(t2)

if __name__ == "__main__":
    main()
