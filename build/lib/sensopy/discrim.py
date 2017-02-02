import numpy as np
# from psychometric_func import *
from scipy.special import binom
from .psychometric_func import PSYCH, GUESSING


class DiscriminationTest(object):
    """"""
    def __init__(self, name=None):
        self.name = name
    
    def psychfunc(self, d):
        raise NotImplementedError
    
    @property
    def guessing(self):
        raise NotImplementedError


class TriangleTest(DiscriminationTest):
    """"""
    def __init__(self):
        super(TriangleTest, self).__init__(name="triangle")
    
    def psychfunc(self, d):
        return PSYCH[self.name](d)
    
    @property
    def guessing(self):
        return GUESSING[self.name]
        

class TwoAFCTest(DiscriminationTest):
    """"""
    def __init__(self):
        super(TwoAFCTest, self).__init__(name="twoAFC")
    
    def psychfunc(self, d):
        return PSYCH[self.name](d)
        
    @property
    def guessing(self):
        return GUESSING[self.name]


class ThreeAFCTest(DiscriminationTest):
    """"""
    def __init__(self):
        super(ThreeAFCTest, self).__init__(name="threeAFC")
    
    def psychfunc(self, d):
        return PSYCH[self.name](d)
    
    @property
    def guessing(self):
        return GUESSING[self.name]
        

class FourAFCTest(DiscriminationTest):
    """"""
    def __init__(self):
        super(FourAFCTest, self).__init__(name="fourAFC")
    
    def psychfunc(self, d):
        return PSYCH[self.name](d)
    
    @property
    def guessing(self):
        return GUESSING[self.name]
        

class MAFCTest(DiscriminationTest):
    """"""
    def __init__(self, m):
        super(MAFCTest, self).__init__(name="{}AFC".format(m))
        self.m = m
    
    def psychfunc(self, d):
        return PSYCH["mAFC"](self.m)(d)
        
    @property
    def guessing(self):
        return 1 / self.m


class TetradUTest(DiscriminationTest):
    """"""
    def __init__(self):
        super(TetradUTest, self).__init__(name="tetu")
    
    def psychfunc(self, d):
        return PSYCH[self.name](d)
        
    @property
    def guessing(self):
        return GUESSING[self.name]
        

class TetradSTest(DiscriminationTest):
    """"""
    def __init__(self):
        super(TetradSTest, self).__init__(name="tets")
    
    def psychfunc(self, d):
        return PSYCH[self.name](d)
    
    @property
    def guessing(self):
        return GUESSING[self.name]
     

class DualPairTest(DiscriminationTest):
    """"""
    def __init__(self):
        super(DualPairTest, self).__init__(name="dualpair")
    
    def psychfunc(self, d):
        return PSYCH[self.name](d)
        
    @property
    def guessing(self):
        return GUESSING[self.name]


class MplusNTest(DiscriminationTest):
    """"""
    def __init__(self, m, n, specified=False):
        super(MplusNTest, self).__init__(name="{}+{}".format(m, n))
        if m < n:
            raise ValueError("Invalid combination of parameters, M and N such that M >= N are expected.")
        self.m = m
        self.m = n
        self.specified = specified
    
    def psychfunc(self, d):
        return None
        
    @property
    def guessing(self):
        if self.m > self.n:
            return 1 / binom(self.m + self.n, self.n)
        elif self.specified:
            return 1 / binom(self.m + self.n, self.n)
        else:
            return 2 / binom(self.m + self.n, self.n)

def main():
    tri      = TriangleTest()
    twoafc   = TwoAFCTest()
    threeafc = ThreeAFCTest()
    fourafc  = FourAFCTest()
    fiveafc  = MAFCTest(5)
    tetu     = TetradUTest()
    tets     = TetradSTest()
    dualpair = DualPairTest()
    
    print(tri.name,      tri.psychfunc(1),      tri.guessing)
    print(twoafc.name,   twoafc.psychfunc(1),   twoafc.guessing)
    print(threeafc.name, threeafc.psychfunc(1), threeafc.guessing)
    print(fourafc.name,  fourafc.psychfunc(1),  fourafc.guessing)
    print(fiveafc.name,  fiveafc.psychfunc(1),  fiveafc.guessing)
    print(tetu.name,     tetu.psychfunc(1),     tetu.guessing)
    print(tets.name,     tets.psychfunc(1),     tets.guessing)
    print(dualpair.name, dualpair.psychfunc(1), dualpair.guessing)
        
if __name__ == "__main__":
    main()
