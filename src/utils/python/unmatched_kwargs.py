import inspect
import functools

#https://stackoverflow.com/a/63787701
def ignore_unmatched_kwargs(f):
    """Make function ignore unmatched kwargs.

    If the function already has the catch all **kwargs, do nothing.
    """
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in inspect.signature(f).parameters.values()):
        return f

    #
    @functools.wraps(f)
    def inner(*args, **kwargs):
        # For each keyword arguments recognised by f,
        # take their binding from **kwargs received
        filtered_kwargs = {
            name: kwargs[name]
            for name, param in inspect.signature(f).parameters.items() if (
                                                                                  param.kind is inspect.Parameter.KEYWORD_ONLY or
                                                                                  param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                                                                          ) and
                                                                          name in kwargs
        }
        return f(*args, **filtered_kwargs)

    return inner