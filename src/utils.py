

def repeat_unary_func(num):
    """
    Decorator that repeats a function of a single input argument `num` times
    E.g.:
    @repeat_unary_func(4)
    def add_1(x):
        return x+1
    """ 
    def wrapper(func):
        def _func(*args, **kwargs):
            assert len(args) + len(kwargs) <= 1, \
                    "Use this decorator only on unary functions"
            res = func(*args, **kwargs)
            for _ in range(num - 1):
                res = func(res)
            return res
        return _func
    return wrapper

