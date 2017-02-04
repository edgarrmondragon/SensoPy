import method


def discrimx(x, n, method, **kwargs):
    pc = x / n
    return METHOD[method](**kwargs)
            
