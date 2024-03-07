from typing import List

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display
from matplotlib import animation


def display_images(images: List[np.ndarray], dpi=100.0, format="html5_video", **kwargs):
    """Display images as an animation in jupyter notebook.

    Args:
        images: images with equal shape.
        dpi: resolution (dots per inch).
        format (str): one of ["html5_video", "jshtml"]

    References:
        https://gist.github.com/foolishflyfox/e30fd8bfbb6a9cee9b1a1fa6144b209c
        http://louistiao.me/posts/notebooks/embedding-matplotlib-animations-in-jupyter-as-interactive-javascript-widgets/
        https://stackoverflow.com/questions/35532498/animation-in-ipython-notebook/46878531#46878531
    """
    h, w = images[0].shape[:2]
    fig = plt.figure(figsize=(h / dpi, w / dpi), dpi=dpi)
    fig_im = plt.figimage(images[0])

    def animate(image):
        fig_im.set_array(image)
        return (fig_im,)

    anim = animation.FuncAnimation(fig, animate, frames=images, **kwargs)
    if format == "html5_video":
        # NOTE(jigu): can not show in VSCode
        display(HTML(anim.to_html5_video()))
    elif format == "jshtml":
        display(HTML(anim.to_jshtml()))
    else:
        raise NotImplementedError(format)

    plt.close(fig)
