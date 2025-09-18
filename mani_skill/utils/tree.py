import torch


# NOTE (stao): when tensordict is used we should replace all of this
def slice(x, i):
    if isinstance(x, dict):
        return {k: slice(v, i) for k, v in x.items()}
    else:
        return x[i]


def cat(x: list):
    if isinstance(x[0], dict):
        return {k: cat([d[k] for d in x]) for k in x[0].keys()}
    else:
        return torch.cat(x, dim=0)


def replace(x, i, y):
    if isinstance(x, dict):
        for k, v in x.items():
            replace(v, i, y[k])
    else:
        x[i] = y

def shape(x, first_only=False):
    """
    Get the shape of leaf items in a tree. If first_only is True, return the shape of the first item only
    """
    if isinstance(x, dict):
        if first_only:
            return shape(next(iter(x.values())), first_only)
        return {k: shape(v, first_only) for k, v in x.items()}
    else:
        return x.shape