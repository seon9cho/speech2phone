# Some helper functions that are generally useful.
# Kyle Roth. 2018-12-12. (my brother's birthday!)

def func_compile(*funcs):
    """Create a function that applies a sequence of functions in order of appearance.
    
    Optionally, parameters may be tuples where the first element is the function and each following element is an
    argument to be passed. 
    """

    def f(obj):
        for func_tup in funcs:
            if hasattr(func_tup, '__call__'):  # if it's actually just a callable, call it
                obj = func_tup(obj)
            elif len(func_tup) == 1:  # otherwise, call the function in the tuple on obj
                obj = func_tup[0](obj)
            else:  # if there are arguments included, pass them in
                obj = func_tup[0](obj, *func_tup[1:])
        return obj
    
    return f
