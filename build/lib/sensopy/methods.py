import numpy as np
import scipy.special
from scipy.integrate import trapz
from scipy.stats import norm


class DiscriminationMethod(object):
    """"""
    def __init__(self, name=None):
        self.name = name

    def psychfunc(self, d):
        """Psychometric function"""
        raise NotImplementedError

    def discriminators(self, d):
        raise NotImplementedError

    @property
    def guessing(self):
        raise NotImplementedError


class Triangle(DiscriminationMethod):
    """Triangle protocol"""
    def __init__(self):
        super(Triangle, self).__init__(name="triangle")

    def psychfunc(self, d):
        """Psychometric function for the Triangle protocol"""
        delta = np.array([d])
        delta = delta.flatten()
        f1 = norm.pdf
        f2 = norm.cdf

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def fi(z):
            return 2 * ((f2(-z * np.sqrt(3) + np.sqrt(2 / 3) * dr) + f2(-z * np.sqrt(3) - np.sqrt(2 / 3) * dr)) * f1(z))

        x = np.linspace(0, 200, 10000)
        y = fi(x)
        i = trapz(y, x)

        return i

    def discriminators(self, d):
        pc = self.psychfunc(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)

    @property
    def guessing(self):
        return 1 / 3


class TwoAFC(DiscriminationMethod):
    """2-AFC protocol"""
    def __init__(self):
        super(TwoAFC, self).__init__(name="twoAFC")

    def psychfunc(self, d):
        """Psychometric function for the 2-AFC protocol"""
        return norm.cdf(d / np.sqrt(2))

    def discriminators(self, d):
        pc = self.psychfunc(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)

    @property
    def guessing(self):
        return 1 / 2


class ThreeAFC(DiscriminationMethod):
    """3-AFC protocol"""

    def __init__(self):
        super(ThreeAFC, self).__init__(name="threeAFC")

    def psychfunc(self, d):
        """Psychometric function for the 3-AFC protocol"""
        delta = np.array([d])
        delta = delta.flatten()

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def fi(z):
            return (norm.cdf(z) ** 2) * norm.pdf(z - dr)

        x = np.linspace(-100, 100, 10000)
        y = fi(x)
        i = trapz(y, x)

        return i

    def discriminators(self, d):
        pc = self.psychfunc(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)

    @property
    def guessing(self):
        return 1 / 3


class FourAFC(DiscriminationMethod):
    """4-AFC protocol"""

    def __init__(self):
        super(FourAFC, self).__init__(name="fourAFC")

    def psychfunc(self, d):
        """Psychometric function for the 4-AFC protocol"""
        delta = np.array([d])
        delta = delta.flatten()

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def fi(z):
            return (norm.cdf(z) ** 3) * norm.pdf(z - dr)

        x = np.linspace(-100, 100, 10000)
        y = fi(x)
        i = trapz(y, x)

        return i

    def discriminators(self, d):
        pc = self.psychfunc(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)

    @property
    def guessing(self):
        return 1 / 4


class MAFC(DiscriminationMethod):
    """m-AFC protocol"""

    def __init__(self, m):
        super(MAFC, self).__init__(name="{}AFC".format(m))
        self.m = m

    def psychfunc(self, d):
        """Psychometric function for the m-AFC protocol"""
        delta = np.array([d])
        delta = delta.flatten()

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def fi(z):
            return (norm.cdf(z) ** (self.m - 1)) * norm.pdf(z - dr)

        x = np.linspace(-100, 100, 10000)
        y = fi(x)
        i = trapz(y, x)

        return i

    def discriminators(self, d):
        pc = self.psychfunc(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)

    @property
    def guessing(self):
        return 1 / self.m


class STetrad(DiscriminationMethod):
    """Specified Tetrad protocol"""

    def __init__(self):
        super(STetrad, self).__init__(name="stetrad")
    
    def psychfunc(self, d):
        """Psychometric function for the Specified Tetrad protocol"""
        delta = np.array([d])
        delta = delta.flatten()
        f1 = norm.pdf
        f2 = norm.cdf

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def fi(z):
            return 2 * (f1(z) * f2(z) * (2 * f2(z - dr) - f2(z - dr) ** 2))

        x = np.linspace(-100, 100, 10000)
        y = fi(x)
        i = trapz(y, x)

        return 1 - i

    def discriminators(self, d):
        pc = self.psychfunc(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)

    @property
    def guessing(self):
        return 1 / 6


class UTetrad(DiscriminationMethod):
    """Unspecified Tetrad protocol"""

    def __init__(self):
        super(UTetrad, self).__init__(name="utetrad")

    def psychfunc(self, d):
        """Psychometric function for the Unspecified Tetrad protocol"""
        delta = np.array([d])
        delta = delta.flatten()
        f1 = norm.pdf
        f2 = norm.cdf

        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d

        def fi(z):
            return 2 * (f1(z) * (2 * f2(z) * f2(z - dr) - f2(z - dr) ** 2))

        x = np.linspace(-100, 100, 10000)
        y = fi(x)
        i = trapz(y, x)

        return 1 - i

    def discriminators(self, d):
        pc = self.psychfunc(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)

    @property
    def guessing(self):
        return 1 / 3


class DualPair(DiscriminationMethod):
    """Dual Pair protocol"""
    def __init__(self):
        super(DualPair, self).__init__(name="dualpair")
    
    def psychfunc(self, d):
        """Psychometric function for the Dual Pair protocol"""
        return norm.cdf(d / 2) ** 2 + norm.cdf(-d / 2) ** 2

    def discriminators(self, d):
        pc = self.psychfunc(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)

    @property
    def guessing(self):
        return 1 / 2


class DuoTrio(DiscriminationMethod):
    """Duo-Trio protocol"""
    def __init__(self):
        super(DuoTrio, self).__init__(name="duotrio")
    
    def psychfunc(self, d):
        """Psychometric function for the Duo-Trio protocol"""
        return 1 - norm.cdf(d / np.sqrt(2)) - norm.cdf(d / np.sqrt(6)) + \
               2 * norm.cdf(d / np.sqrt(2)) * norm.cdf(d / np.sqrt(6))

    def discriminators(self, d):
        pc = self.psychfunc(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)

    @property
    def guessing(self):
        return 1 / 2


class MplusN(DiscriminationMethod):
    """"""
    def __init__(self, m, n, specified=False):
        if specified:
            c = "S"
        else:
            c = "U"
        super(MplusN, self).__init__(name="{}+{}({})".format(m, n, c))
        if m < n:
            raise ValueError("Invalid combination of parameters. M >= N expected.")
        self.m = m
        self.n = n
        self.specified = specified
    
    def psychfunc(self, d):
        """TODO"""
        raise NotImplementedError

    def discriminators(self, d):
        pc = self.psychfunc(d)
        pg = self.guessing
        return (pc - pg) / (1 - pg)

    @property
    def guessing(self):
        if self.m > self.n:
            return 1 / scipy.special.binom(self.m + self.n, self.n)
        elif self.specified:
            return 1 / scipy.special.binom(self.m + self.n, self.n)
        else:
            return 2 / scipy.special.binom(self.m + self.n, self.n)


METHOD = dict(triangle=Triangle,
              two_afc=TwoAFC,
              three_afc=ThreeAFC,
              four_afc=FourAFC,
              duotrio=DuoTrio,
              dualpair=DualPair,
              utetrad=UTetrad,
              stetrad=STetrad,
              m_afc=MAFC)


def main():

    m1 = Triangle()
    print(m1.psychfunc.__doc__)
    print(isinstance(m1, DiscriminationMethod))
    
    s = "%10s  %1.5f  %1.5f"
    delta = 0
    
    for method in ("triangle", "two_afc", "three_afc", "utetrad", "stetrad", "m_afc"):
        if method == "m_afc":
            d1 = discrimx(11, 24, method, m=5)
        else:
            d1 = discrimx(11, 24, method)
        print(s % (d1.name, d1.guessing, d1.psychfunc(delta)), d1.psychfunc.__doc__)
    
        
if __name__ == "__main__":
    main()
