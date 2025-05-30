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
