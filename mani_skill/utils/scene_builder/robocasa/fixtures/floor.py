from mani_skill.envs.scene import ManiSkillScene


class Floor:
    def __init__(self, scene: ManiSkillScene):
        self.scene = scene

    def build(self):
        # builder = self.scene.create_actor_builder()
        # builder.add_box_collision(half_size=[10, 10, 0.01])
        # self.actor = builder.build_static(name="floor")
        return self
