SAPIEN_RENDER_SYSTEM = "3.0"
try:
    # NOTE (stao): hacky way to determine which render system in sapien 3 is being used for testing purposes
    from sapien.wrapper.scene import get_camera_shader_pack

    SAPIEN_RENDER_SYSTEM = "3.1"
except:
    pass


class RenderConfig:
    """class for working with SAPIEN shader system"""

    def __init__(self, shader_pack: str, obs_mode: str):
        self.shader_pack = shader_pack


class MinimalRenderConfig(RenderConfig):
    def __init__(self):
        super().__init__(shader_pack="minimal")


# SHADER_CONFIGS = {
#     "minimal": ShaderConfig(shader_pack="minimal"),
#     "rt": ShaderConfig(shader_pack="rt"),
# }
