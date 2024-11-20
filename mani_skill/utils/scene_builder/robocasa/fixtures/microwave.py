import numpy as np

from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import Fixture


class Microwave(Fixture):
    """
    Microwave fixture class. Supports turning on and off the microwave, and opening and closing the door

    Args:
        xml (str): path to mjcf xml file

        name (str): name of the object
    """

    def __init__(
        self,
        xml="fixtures/microwaves/orig_microwave",
        name="microwave",
        *args,
        **kwargs,
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )
        self._turned_on = False

    def set_door_state(self, min, max, env, rng):
        """
        Sets how open the door is. Chooses a random amount between min and max.
        Min and max are percentages of how open the door is

        Args:
            min (float): minimum percentage of how open the door is

            max (float): maximum percentage of how open the door is

            env (MujocoEnv): environment

            rng (np.random.Generator): random number generator

        """
        assert 0 <= min <= 1 and 0 <= max <= 1 and min <= max

        joint_min = 0
        joint_max = np.pi / 2

        desired_min = joint_min + (joint_max - joint_min) * min
        desired_max = joint_min + (joint_max - joint_min) * max

        sign = -1

        env.sim.data.set_joint_qpos(
            "{}_microjoint".format(self.name),
            sign * rng.uniform(desired_min, desired_max),
        )

    def get_door_state(self, env):
        """
        Args:
            env (MujocoEnv): environment

        Returns:
            dict: maps door name to a percentage of how open the door is
        """
        sim = env.sim
        hinge_qpos = sim.data.qpos[sim.model.joint_name2id(f"{self.name}_microjoint")]
        hinge_qpos = -hinge_qpos  # negate as micro joints are left door hinges

        # convert to percentages
        door = OU.normalize_joint_value(hinge_qpos, joint_min=0, joint_max=np.pi / 2)

        return {
            "door": door,
        }

    def get_state(self):
        """
        Returns:
            dict: maps turned_on to whether the microwave is turned on
        """
        state = dict(
            turned_on=self._turned_on,
        )
        return state

    @property
    def handle_name(self):
        return "{}_door_handle".format(self.name)

    @property
    def door_name(self):
        return "{}_door".format(self.name)

    def update_state(self, env):
        """
        If the microwave is open, the state is set to off. Otherwise, if the gripper
        is pressing the start button, the microwave will stay/turn on. If the gripper is
        pressing the stop button, the microwave will stay/turn off.

        Args:
            env (MujocoEnv): The environment to check the state of the microwave in

        """
        start_button_pressed = env.check_contact(
            env.robots[0].gripper["right"], "{}_start_button".format(self.name)
        )
        stop_button_pressed = env.check_contact(
            env.robots[0].gripper["right"], "{}_stop_button".format(self.name)
        )

        door_state = self.get_door_state(env)["door"]
        door_open = door_state > 0.005

        if door_open:
            self._turned_on = False
        else:
            if self._turned_on is True and stop_button_pressed:
                self._turned_on = False
            elif self._turned_on is False and start_button_pressed:
                self._turned_on = True

    def gripper_button_far(self, env, button, th=0.15):
        """
        check whether gripper is far from the start button

        Args:
            env (MujocoEnv): The environment to check the state of the microwave in

            button (str): button to check

            th (float): threshold for distance between gripper and button

        Returns:
            bool: True if gripper is far from the button, False otherwise
        """
        assert button in ["start_button", "stop_button"]
        button_id = env.sim.model.geom_name2id(
            "{}{}".format(self.naming_prefix, button)
        )
        button_pos = env.sim.data.geom_xpos[button_id]
        gripper_site_pos = env.sim.data.site_xpos[env.robots[0].eef_site_id["right"]]

        gripper_button_far = np.linalg.norm(gripper_site_pos - button_pos) > th

        return gripper_button_far

    @property
    def nat_lang(self):
        return "microwave"
