import platform
from typing import Union

import sapien


def can_render(device: Union[sapien.Device, None]) -> bool:
    """Whether or not this device can render, depending on the rendering device selected"""
    # NOTE (stao): currently sapien can't tell if the render device can render or not for MacOS
    if platform.system() == "Darwin":
        return True
    # NOTE (stao): sapien's can_render function is not always accurate. The alternative at the moment is to let the user
    # try to render and if there is a bug, tell the user to disable rendering by setting render_backend to "none" or None.
    return device is not None
