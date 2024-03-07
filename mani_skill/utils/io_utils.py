import gzip
import json
from pathlib import Path
from typing import Sequence, Union

import numpy as np


class CustomJsonEncoder(json.JSONEncoder):
    """Custom json encoder to support more types, like numpy and Path."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def load_json(filename: Union[str, Path]):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "rt")
    elif filename.endswith(".json"):
        f = open(filename, "rt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    ret = json.loads(f.read())
    f.close()
    return ret


def dump_json(filename: Union[str, Path], obj, **kwargs):
    filename = str(filename)
    if filename.endswith(".gz"):
        f = gzip.open(filename, "wt")
    elif filename.endswith(".json"):
        f = open(filename, "wt")
    else:
        raise RuntimeError(f"Unsupported extension: {filename}")
    json.dump(obj, f, cls=CustomJsonEncoder, **kwargs)
    f.close()


def write_txt(filename: Union[str, Path], content: Union[str, Sequence[str]]):
    with open(filename, "w") as f:
        if not isinstance(content, str):
            content = "\n".join(content)
        f.write(content)
