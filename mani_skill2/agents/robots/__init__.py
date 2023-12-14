from .mobile_panda import MobilePandaDualArm, MobilePandaSingleArm
from .panda import Panda
from .xmate3 import Xmate3Robotiq

ROBOTS = {
    "panda": Panda,
    "mobile_panda_dual_arm": MobilePandaDualArm,
    "mobile_panda_single_arm": MobilePandaSingleArm,
    "xmate3_robotiq": Xmate3Robotiq,
}
