import methods


def discrimx(x, n, method, **kwargs):
    pc = x / n
    
    
    
    return methods.METHOD[method](**kwargs)
            
def main():
    m1 = methods.Triangle()
    print(m1.psychfunc.__doc__)
    print(isinstance(m1, methods.DiscriminationMethod))
    
    s = "%10s  %1.5f  %1.5f"
    delta = 1.434
    
    for method in ("triangle", "two_afc", "three_afc", "utetrad", "stetrad", "m_afc"):
        if method == "m_afc":
            d1 = discrimx(11, 24, method, m=5)
        else:
            d1 = discrimx(11, 24, method)
        print(s % (d1.name, d1.guessing, d1.psychfunc(delta)), d1.psychfunc.__doc__)
        
    m2 = methods.MplusN(2, 2)
    print(m2.psychfunc.__doc__)
    print(m2.name)
    print(m2.psychfunc(delta))
    print(m2.psychfunc(1.01))
    print(m2.psychfunc(1.02))
    print(m2.psychfunc(1.03))
    print(m2.psychfunc(1.04))
    print(m2.psychfunc(1.05))

if __name__ == "__main__":
    main()
