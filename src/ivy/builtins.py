def builtin_range(*args, bound=None):
    if len(args) == 2:
        start, stop = args
    else:
        start, stop = 0, args[0]

    if bound:
        if stop > bound and len(args) == 1:
            raise RuntimeError(f"Stop value is greater than bound={bound} value")
        if stop - start > bound:
            raise RuntimeError(f"Range is greater than bound={bound} value")

    return range(start, stop)


def builtin_len(x):
    return len(x)


def builtin_print(*args):
    print(*args)
