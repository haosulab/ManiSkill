import platform

import sapien


def can_render(device: sapien.Device) -> bool:
    """Whether or not this device can render, depending on the rendering device selected"""
    # NOTE (stao): currently sapien can't tell if the render device can render or not for MacOS
    if platform.system() == "Darwin":
        return True
    return device.can_render()
