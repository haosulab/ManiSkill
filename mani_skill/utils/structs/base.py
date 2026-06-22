from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, TypeVar, cast

import torch

from mani_skill.utils import common

if TYPE_CHECKING:
    from mani_skill.sim.base_sim import BaseSim
T = TypeVar("T")


@dataclass(kw_only=True)
class BaseStruct:
    """
    Base class of all structs that manage objects in simulation across sub-scenes.
    """

    _scene_idxs: torch.Tensor
    """A list of indexes indicating which sub-scene each managed object is in."""
    sim: BaseSim
    """
    The simulation backend for this struct.
    """

    def __post_init__(self):
        if not isinstance(self._scene_idxs, torch.Tensor):
            self._scene_idxs = common.to_tensor(cast(list[int], self._scene_idxs))
        self._scene_idxs = self._scene_idxs.to(torch.int).to(self.device)

    def __str__(self):
        return f"<struct of type {self.__class__}; managing {self._num_objs} objects>"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__maniskill_hash__

    @cached_property
    def __maniskill_hash__(self):
        """A better hash to use compared to the default frozen dataclass hash.
        It is tied directly to the only immutable field (the _objs list)."""
        return hash(tuple([obj.__hash__() for obj in self._objs]))

    @property
    def device(self):
        """The device that simulation data is returned on."""
        # TODO (stao): split between sim and render device? One can check more accurately via
        # which render and physics sim is used.
        return self.sim.sim_device_torch

    @property
    def _num_objs(self):
        return self._scene_idxs.shape[0]
