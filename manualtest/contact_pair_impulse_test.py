import numpy as np
import sapien
from torch import Tensor

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.building.actors import build_actor_ycb
from mani_skill.utils.scene_builder.table.table_scene_builder import TableSceneBuilder
from mani_skill.utils.structs import Actor


class GraspBerry(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(self)
        self.table_scene.build()
        self.obj: Actor = build_actor_ycb("012_strawberry", self._scene, name="obj")[0]

    def _initialize_episode(self, env_idx: Tensor):
        self.table_scene.initialize(env_idx)

        qpos = np.array(
            [
                0.061308865,
                0.6911312,
                0.037140213,
                -1.9521213,
                0.030660806,
                2.6812325,
                0.892481,
                0.035882425,
                0.035938274,
            ]
        )
        self.agent.robot.set_qpos(qpos)
        self.obj.set_pose(
            sapien.Pose(
                [-0.00650965, 0.0598809, 0.0221112],
                [0.66107, 0.0558751, -0.092696, 0.742484],
            )
        )

    def evaluate(self) -> dict:
        lf = self.agent.robot.get_net_contact_forces(["panda_leftfinger"])
        rf = self.agent.robot.get_net_contact_forces(["panda_rightfinger"])
        print("LF", lf)
        print("RF", rf)
        agent: Panda = self.agent
        success = self.agent.is_grasping(self.obj)
        impulses = (
            agent.queries[self.obj.name][0]
            .cuda_impulses.torch()
            .clone()
            .reshape(2, -1, 3)
        )
        print("Left Impulse", impulses[0, ...])
        print("Right Impulse", impulses[1, ...])
        return {"success": success}


def main():
    env = GraspBerry(num_envs=1, force_use_gpu_sim=True, reward_mode="sparse")
    env.reset(seed=0)
    i = 0
    while True:
        action = env.action_space.sample()
        action *= 0
        action[-1] = np.sin(i / 10)
        env.step(action)
        env.render_human()
        i += 1


if __name__ == "__main__":
    main()
