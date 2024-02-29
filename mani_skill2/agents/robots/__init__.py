from mani_skill2.agents.robots.anymal.anymal_c import ANYmalC
from .dclaw import DClaw
from .fetch import Fetch
from .mobile_panda import MobilePandaDualArm, MobilePandaSingleArm
from .panda import Panda
from .xarm import XArm7Ability
from .xmate3 import Xmate3Robotiq
from .allegro_hand import AllegroHandRightTouch, AllegroHandRight, AllegroHandLeft

ROBOTS = {
    "panda": Panda,
    "mobile_panda_dual_arm": MobilePandaDualArm,
    "mobile_panda_single_arm": MobilePandaSingleArm,
    "xmate3_robotiq": Xmate3Robotiq,
    "fetch": Fetch,
    # Dexterous Hand
    "dclaw": DClaw,
    "xarm7_ability": XArm7Ability,
    "allegro_hand_right": AllegroHandRight,
    "allegro_hand_left": AllegroHandLeft,
    "allegro_hand_right_touch": AllegroHandRightTouch,
    # Locomotion
    "anymal-c": ANYmalC,
}
