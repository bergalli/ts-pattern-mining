from copy import deepcopy
from typing import Callable


def immutable_fun(fun):
    def dec(*args, **kwargs):
        args = tuple(map(deepcopy, args))
        kwargs = {k: deepcopy(v) for k, v in kwargs.items()}

        output = fun(*args, **kwargs)

        return output

    return dec


def log_running_time(func: Callable) -> Callable:
    """Decorator for logging node execution time.
        Args:
            func: Function to be executed.
        Returns:
            Decorator for logging the running time.
    """

    @wraps(func)
    def with_time(*args, **kwargs):
        log = logging.getLogger(__name__)
        t_start = time.time()
        result = func(*args, **kwargs)
        t_end = time.time()
        elapsed = t_end - t_start
        log.info("Running %r took %.2f seconds", func.__name__, elapsed)
        return result

    return with_time
