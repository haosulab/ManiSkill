import sapien
import sapien.physx as physx

from mani_skill.sim.base_system import BaseSystem
from mani_skill.sim.mpm.config import MPMSystemConfig


class MPMSystem(BaseSystem):
    """
    MPM system
    """

    def __init__(self):
        super().__init__()
        self.config = MPMSystemConfig()

    def add_boundaries_by_sapien_scene(self, scene: sapien.Scene):
        """
        Add boundaries to the MPM system based on the given sapien scene
        """

    def add_boundaries_from_sapien_physx_rigid_body_component(
        self, rigid_body_component: physx.PhysxRigidBodyComponent
    ):
        """
        Add boundaries to the MPM system based on the given sapien physx rigid body component
        """
        for shape in rigid_body_component.collision_shapes:
            # do something
            pass

    def init(self):
        """
        Initialize the MPM system based on the given sapien rigid body components to serve as boundaries
        """

    def step(self):
        pass

    def set_config(self, config: MPMSystemConfig):
        self.config = config
