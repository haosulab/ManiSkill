from mani_skill2 import PACKAGE_ASSET_DIR
from mani_skill2.agents.base_agent import BaseAgent
from mani_skill2.agents.controllers import *


class ANYmalC(BaseAgent):
    uid = "anymal-c"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/anymal-c/urdf/anymal.urdf"
    urdf_config = dict()

    def __init__(self, *args, **kwargs):
        self.joint_names = [
            "LF_HAA",
            "RF_HAA",
            "LH_HAA",
            "RH_HAA",
            "LF_HFE",
            "RF_HFE",
            "LH_HFE",
            "RH_HFE",
            "LF_KFE",
            "RF_KFE",
            "LH_KFE",
            "RH_KFE",
        ]
        super().__init__(*args, **kwargs)

    @property
    def controller_configs(self):
        self._load_articulation()
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100
        # import ipdb;ipdb.set_trace()
        pd_joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=True,
            use_delta=True,
        )
        controller_configs = dict(pd_joint_delta_pos=pd_joint_delta_pos)
        return controller_configs

    def _after_init(self):
        pass

    sensor_configs = []
