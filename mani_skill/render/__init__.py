from contextlib import contextmanager

import sapien

SAPIEN_RENDER_SYSTEM = "3.0"
try:
    # NOTE (stao): hacky way to determine which render system in sapien 3 is being used for testing purposes
    from sapien.wrapper.scene import get_camera_shader_pack

    SAPIEN_RENDER_SYSTEM = "3.1"
except:
    pass

GlobalShaderPack = None


@contextmanager
def set_shader_pack(shader_pack):
    global GlobalShaderPack
    old = GlobalShaderPack
    GlobalShaderPack = shader_pack
    yield
    GlobalShaderPack = old
