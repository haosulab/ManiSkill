"""
Code partly adapted from https://github.com/google/flax/blob/main/flax/struct.py

Changed to maybe support torch and jax backends for the purposes of ManiSkill
"""

import dataclasses
from typing import TypeVar

from typing_extensions import dataclass_transform  # pytype: disable=not-supported-yet

_T = TypeVar("_T")


def field(pytree_node=True, **kwargs):
    return dataclasses.field(metadata={"pytree_node": pytree_node}, **kwargs)


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def dataclass(clz: _T) -> _T:
    # check if already a maniskill dataclass
    if "_ms_dataclass" in clz.__dict__:
        return clz

    data_clz = dataclasses.dataclass(frozen=True)(clz)  # type: ignore
    meta_fields = []
    data_fields = []
    for field_info in dataclasses.fields(data_clz):
        is_pytree_node = field_info.metadata.get("pytree_node", True)
        if is_pytree_node:
            data_fields.append(field_info.name)
        else:
            meta_fields.append(field_info.name)

    def replace(self, **updates):
        """ "Returns a new object replacing the specified fields with new values."""
        return dataclasses.replace(self, **updates)

    data_clz.replace = replace

    def iterate_clz(x):
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta

    # def iterate_clz_with_keys(x):
    #     meta = tuple(getattr(x, name) for name in meta_fields)
    #     data = tuple(
    #     (jax.tree_util.GetAttrKey(name), getattr(x, name)) for name in data_fields
    #     )
    #     return data, meta

    # def clz_from_iterable(meta, data):
    #     meta_args = tuple(zip(meta_fields, meta))
    #     data_args = tuple(zip(data_fields, data))
    #     kwargs = dict(meta_args + data_args)
    #     return data_clz(**kwargs)

    #   jax.tree_util.register_pytree_with_keys(
    #     data_clz, iterate_clz_with_keys, clz_from_iterable, iterate_clz,
    #   )

    #   def to_state_dict(x):
    #     state_dict = {
    #       name: serialization.to_state_dict(getattr(x, name))
    #       for name in data_fields
    #     }
    #     return state_dict

    #   def from_state_dict(x, state):
    #     """Restore the state of a data class."""
    #     state = state.copy()  # copy the state so we can pop the restored fields.
    #     updates = {}
    #     for name in data_fields:
    #       if name not in state:
    #         raise ValueError(
    #           f'Missing field {name} in state dict while restoring'
    #           f' an instance of {clz.__name__},'
    #           f' at path {serialization.current_path()}'
    #         )
    #       value = getattr(x, name)
    #       value_state = state.pop(name)
    #       updates[name] = serialization.from_state_dict(
    #         value, value_state, name=name
    #       )
    #     if state:
    #       names = ','.join(state.keys())
    #       raise ValueError(
    #         f'Unknown field(s) "{names}" in state dict while'
    #         f' restoring an instance of {clz.__name__}'
    #         f' at path {serialization.current_path()}'
    #       )
    #     return x.replace(**updates)

    #   serialization.register_serialization_state(
    #     data_clz, to_state_dict, from_state_dict
    #   )

    # add a _ms_dataclass flag to distinguish from regular dataclasses
    data_clz._ms_dataclass = True  # type: ignore[attr-defined]

    return data_clz  # type: ignore


TNode = TypeVar("TNode", bound="PyTreeNode")


@dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
class PyTreeNode:
    def __init_subclass__(cls):
        dataclass(cls)  # pytype: disable=wrong-arg-types

    def __init__(self, *args, **kwargs):
        # stub for pytype
        raise NotImplementedError

    def replace(self: TNode, **overrides) -> TNode:
        # stub for pytype
        raise NotImplementedError
