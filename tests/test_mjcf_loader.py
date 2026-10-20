import sapien
import sapien.physx as physx
import pytest

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.envs.utils.system.backend import parse_sim_and_render_backend
from mani_skill.utils.building.mjcf_loader import MJCFLoader

ARM_WITH_EQUALITY_MJCF = """<mujoco model="test_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="box" size="0.05 0.05 0.05"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="j1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="j2" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
  <equality>
    <joint joint1="j2" joint2="j1" polycoef="0.1 2 0 0 0"/>
  </equality>
</mujoco>
"""

ARM_WITH_FIXED_TENDON_MJCF = """<mujoco model="test_tendon">
  <worldbody>
    <body name="b1">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="b2">
        <joint name="j2" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="t1">
      <joint joint="j1" coef="0.5"/>
      <joint joint="j2" coef="-0.5"/>
    </fixed>
  </tendon>
</mujoco>
"""


@pytest.fixture
def loader():
    # a CPU scene is only needed so the loader can create builders through it
    backend = parse_sim_and_render_backend("cpu", "cpu")
    sub_scenes = [sapien.Scene(systems=[physx.PhysxCpuSystem()])]
    scene = ManiSkillScene(sub_scenes=sub_scenes, device=backend.device, backend=backend)
    loader = MJCFLoader()
    loader.set_scene(scene)
    return loader


def test_mjcf_equality_joint(tmp_path, loader):
    mjcf_file = tmp_path / "arm.xml"
    mjcf_file.write_text(ARM_WITH_EQUALITY_MJCF)
    data = loader.parse(str(mjcf_file))
    assert len(data["articulation_builders"]) == 1
    records = data["articulation_builders"][0].mimic_joint_records
    assert len(records) == 1
    record = records[0]
    # MuJoCo joint equality constrains q1 = a0 + a1 * q2 which maps to a mimic joint
    assert record.joint == "j2"
    assert record.mimic == "j1"
    assert record.multiplier == pytest.approx(2.0)
    assert record.offset == pytest.approx(0.1)


def test_mjcf_fixed_tendon(tmp_path, loader):
    mjcf_file = tmp_path / "tendon.xml"
    mjcf_file.write_text(ARM_WITH_FIXED_TENDON_MJCF)
    data = loader.parse(str(mjcf_file))
    records = data["articulation_builders"][0].mimic_joint_records
    assert len(records) == 1
    record = records[0]
    # 0.5 * q1 - 0.5 * q2 = 0  =>  q1 = q2
    assert record.joint == "j1"
    assert record.mimic == "j2"
    assert record.multiplier == pytest.approx(1.0)
    assert record.offset == pytest.approx(0.0)
