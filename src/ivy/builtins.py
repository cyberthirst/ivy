def builtin_range(start_or_stop, stop=None, step=None, bound=None):
    if stop is None:
        start, stop = 0, start_or_stop
    else:
        start = start_or_stop

    if step is None:
        step = 1

    if bound is not None:
        stop = min(stop, bound)

    # Ensure all arguments are integers
    # start, stop, step = int(start), int(stop), int(step)

    # Runtime checks
    if stop <= start:
        raise ValueError("STOP must be greater than START")
    if step <= 0:
        raise ValueError("Step must be positive")

    return range(start, stop, step)


def builtin_len(x):
    return len(x)


def builtin_print(*args):
    print(*args)
