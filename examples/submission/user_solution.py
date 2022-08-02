from mani_skill2.evaluation.solution import BasePolicy


class UserPolicy(BasePolicy):
    def act(self, observations):
        return self.action_space.sample()

    @classmethod
    def get_obs_mode(cls, env_id: str) -> str:
        return "rgbd"

    @classmethod
    def get_control_mode(cls, env_id: str) -> str:
        return "pd_joint_delta_pos"
