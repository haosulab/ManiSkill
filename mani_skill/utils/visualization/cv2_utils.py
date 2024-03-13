import os
from typing import List

import cv2
import numpy as np
import tqdm


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    verbose: bool = True,
    is_rgb=True,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    output_path = os.path.join(output_dir, video_name)
    image_shape = images[0].shape
    frame_size = (image_shape[1], image_shape[0])
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
    if verbose:
        print(f"Video created: {output_path}")
        images_iter = tqdm.tqdm(images)
    else:
        images_iter = images
    for im in images_iter:
        im = im[..., 0:3]
        if is_rgb:
            im = im[..., ::-1]
        writer.write(im)
    writer.release()


class OpenCVViewer:
    def __init__(self, name="OpenCVViewer", is_rgb=True, exit_on_esc=True):
        self.name = name
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        self.is_rgb = is_rgb
        self.exit_on_esc = exit_on_esc

    def imshow(self, image: np.ndarray, is_rgb=None, non_blocking=False, delay=0):
        if image.ndim == 2:
            image = np.tile(image[..., np.newaxis], (1, 1, 3))
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        assert image.ndim == 3, image.shape

        if self.is_rgb or is_rgb:
            image = image[..., ::-1]
        cv2.imshow(self.name, image)

        if non_blocking:
            return
        else:
            key = cv2.waitKey(delay)
            if key == 27:  # escape
                if self.exit_on_esc:
                    exit(0)
                else:
                    return None
            elif key == -1:  # timeout
                pass
            else:
                return chr(key)

    def close(self):
        cv2.destroyWindow(self.name)

    def __del__(self):
        self.close()
