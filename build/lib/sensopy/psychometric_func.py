import numpy as np
from scipy.integrate import trapz
from scipy.stats import norm

pf_twoAFC = lambda d: norm.cdf(d / np.sqrt(2))

pf_duotrio = lambda d: 1 - norm.cdf(d / np.sqrt(2)) - norm.cdf(d / np.sqrt(6)) +\
                       2 * norm.cdf(d / np.sqrt(2)) * norm.cdf(d / np.sqrt(6))

def pf_threeAFC(d):
    delta = np.array([d])
    delta = delta.flatten()
    
    if len(delta) > 1:
        dr = np.reshape(delta, (len(delta), 1))
    else:
        dr = d
    
    f1 = lambda z: norm.pdf(z)
    f2 = lambda z: norm.cdf(z)
    fi = lambda z: (f2(z) ** 2) * f1(z - dr)
    x = np.linspace(-10, 10, 10000)
    y = fi(x)
    i = trapz(y, x)

    return i
    
def pf_fourAFC(d):
    delta = np.array([d])
    delta = delta.flatten()
    
    if len(delta) > 1:
        dr = np.reshape(delta, (len(delta), 1))
    else:
        dr = d
    
    f1 = lambda z: norm.pdf(z)
    f2 = lambda z: norm.cdf(z)
    fi = lambda z: (f2(z) ** 3) * f1(z - dr)
    x = np.linspace(-10, 10, 10000)
    y = fi(x)
    i = trapz(y, x)

    return i

def pf_mAFC(m):
    def pf(d):
        delta = np.array([d])
        delta = delta.flatten()
        
        if len(delta) > 1:
            dr = np.reshape(delta, (len(delta), 1))
        else:
            dr = d
        
        f1 = lambda z: norm.pdf(z)
        f2 = lambda z: norm.cdf(z)
        fi = lambda z: (f2(z) ** (m - 1)) * f1(z - dr)
        x = np.linspace(-10, 10, 10000)
        y = fi(x)
        i = trapz(y, x)
    
        return i
    return pf
    
def pf_triangle(d):
    delta = np.array([d])
    delta = delta.flatten()
    f1 = lambda z: norm.pdf(z)
    f2 = lambda z: norm.cdf(z)
    
    if len(delta) > 1:
        dr = np.reshape(delta, (len(delta), 1))
    else:
        dr = d
    
    fi = lambda z: 2 * ((f2(-z * np.sqrt(3) + np.sqrt(2 / 3) * dr) + f2(-z * np.sqrt(3) - \
                   np.sqrt(2 / 3) * dr)) * f1(z))
    
    x = np.linspace(0, 10, 10000)
    y = fi(x)
    i = trapz(y, x)
    
    return i
    
def pf_tetu(d):
    delta = np.array([d])
    delta = delta.flatten()
    f1 = lambda z: norm.pdf(z)
    f2 = lambda z: norm.cdf(z)
    
    if len(delta) > 1:
        dr = np.reshape(delta, (len(delta), 1))
    else:
        dr = d
    
    fi = lambda z: 2 * (f1(z) * (2 * f2(z) * f2(z - dr) - f2(z - dr) ** 2))
    
    x = np.linspace(-10, 10, 10000)
    y = fi(x)
    i = trapz(y, x)
    
    return 1 - i
    
def pf_tets(d):
    delta = np.array([d])
    delta = delta.flatten()
    f1 = lambda z: norm.pdf(z)
    f2 = lambda z: norm.cdf(z)
    
    if len(delta) > 1:
        dr = np.reshape(delta, (len(delta), 1))
    else:
        dr = d
    
    fi = lambda z: 2 * (f1(z) * f2(z) * (2 * f2(z - dr) - f2(z - dr) ** 2))
    
    x = np.linspace(-10, 10, 10000)
    y = fi(x)
    i = trapz(y, x)
    
    return 1 - i

pf_dualpair = lambda d: norm.cdf(d / 2) ** 2 + norm.cdf(-d / 2) ** 2


TEST = [
        "triangle",
        "twoAFC",
        "threeAFC",
        "fourAFC",
        "tetu",
        "tets",
        "duotrio",
        "dualpair"
]

PSYCH = {  
            "triangle": pf_triangle,
            "twoAFC":   pf_twoAFC,
            "threeAFC": pf_threeAFC,
            "fourAFC":  pf_fourAFC,
            "mAFC":     pf_mAFC,
            "tetu":     pf_tetu,
            "tets":     pf_tets,
            "duotrio":  pf_duotrio,
            "dualpair": pf_dualpair
}

GUESSING = {
            "triangle": 1 / 3,
            "twoAFC":   1 / 2,
            "threeAFC": 1 / 3,
            "fourAFC":  1 / 4,
            "tetu":     1 / 3,
            "tets":     1 / 6,
            "duotrio":  1 / 2,
            "dualpair": 1 / 2
}


def main():
    pass

if __name__ == "__main__":
    main()
