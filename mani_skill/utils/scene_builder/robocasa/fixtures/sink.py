from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import Fixture


class Sink(Fixture):
    def __init__(
        self,
        scene: ManiSkillScene,
        xml="fixtures/sink.xml",
        name="sink",
        *args,
        **kwargs
    ):
        self._handle_joint = None
        self._water_site = None
        super().__init__(
            scene=scene,
            xml=xml,
            name=name,
            duplicate_collision_geoms=False,
            *args,
            **kwargs
        )

    # TODO (stao): Add fake water
    def update_state(self, env):
        """
        Updates the water flowing of the sink based on the handle_joint position

        Args:
            env (MujocoEnv): environment
        """
        state = self.get_handle_state(env)
        water_on = state["water_on"]

        site_id = env.sim.model.site_name2id("{}water".format(self.naming_prefix))

        if water_on:
            env.sim.model.site_rgba[site_id][3] = 0.5
        else:
            env.sim.model.site_rgba[site_id][3] = 0.0

    def set_handle_state(self, env, rng, mode="on"):
        """
        Sets the state of the handle_joint based on the mode parameter

        Args:
            env (MujocoEnv): environment

            rng (np.random.Generator): random number generator

            mode (str): "on", "off", or "random"
        """
        assert mode in ["on", "off", "random"]
        if mode == "random":
            mode = rng.choice(["on", "off"])

        if mode == "off":
            joint_val = 0.0
        elif mode == "on":
            joint_val = rng.uniform(0.40, 0.50)

        env.sim.data.set_joint_qpos(
            "{}handle_joint".format(self.naming_prefix), joint_val
        )

    def get_handle_state(self, env):
        """
        Gets the state of the handle_joint

        Args:
            env (MujocoEnv): environment

        Returns:
            dict: maps handle_joint to the angle of the handle_joint, water_on to whether the water is flowing,
            spout_joint to the angle of the spout_joint, and spout_ori to the orientation of the spout (left, right, center)
        """
        handle_state = {}
        if self.handle_joint is None:
            return handle_state

        handle_joint_id = env.sim.model.joint_name2id(
            "{}handle_joint".format(self.naming_prefix)
        )
        handle_joint_qpos = deepcopy(env.sim.data.qpos[handle_joint_id])
        handle_joint_qpos = handle_joint_qpos % (2 * np.pi)
        if handle_joint_qpos < 0:
            handle_joint_qpos += 2 * np.pi
        handle_state["handle_joint"] = handle_joint_qpos
        handle_state["water_on"] = 0.40 < handle_joint_qpos < np.pi

        spout_joint_id = env.sim.model.joint_name2id(
            "{}spout_joint".format(self.naming_prefix)
        )
        spout_joint_qpos = deepcopy(env.sim.data.qpos[spout_joint_id])
        spout_joint_qpos = spout_joint_qpos % (2 * np.pi)
        if spout_joint_qpos < 0:
            spout_joint_qpos += 2 * np.pi
        handle_state["spout_joint"] = spout_joint_qpos
        if np.pi <= spout_joint_qpos <= 2 * np.pi - np.pi / 6:
            spout_ori = "left"
        elif np.pi / 6 <= spout_joint_qpos <= np.pi:
            spout_ori = "right"
        else:
            spout_ori = "center"
        handle_state["spout_ori"] = spout_ori

        return handle_state

    @property
    def handle_joint(self):
        """
        Returns the joint element which represents the handle_joint of the sink
        """
        if self._handle_joint is None:
            self._handle_joint = self.worldbody.find(
                "./body/body/body/joint[@name='{}handle_joint']".format(
                    self.naming_prefix
                )
            )

        return self._handle_joint

    @property
    def water_site(self):
        """
        Returns the site element which represents the water flow of the sink
        """
        if self._water_site is None:
            self._water_site = self.worldbody.find(
                "./body/body/body/site[@name='{}water']".format(self.naming_prefix)
            )

        return self._water_site

    @property
    def nat_lang(self):
        return "sink"
