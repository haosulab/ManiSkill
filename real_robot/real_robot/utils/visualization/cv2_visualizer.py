from typing import List

import numpy as np
import cv2


class CV2Visualizer:
    def __init__(self, window_name="Images"):
        self.window_name = window_name
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    @staticmethod
    def preprocess_image(image: np.ndarray, depth_scale=1000.0) -> np.ndarray:
        """Preprocess image for plotting with cv2
        :param image: depth or RGB color image
        :return image: color image in BGR format
        """
        channels = image.shape[-1]
        if image.ndim == 2 or (image.ndim == 3 and channels == 1):
            # Depth image colormap is taken from
            # https://github.com/IntelRealSense/librealsense/blob/8ffb17b027e100c2a14fa21f01f97a1921ec1e1b/wrappers/python/examples/opencv_viewer_example.py#L56
            if image.dtype == np.uint16:
                alpha = 0.03
            elif np.issubdtype(image.dtype, np.floating):
                alpha = 0.03 * depth_scale
            else:
                raise TypeError(f"Unknown image dtype: {image.dtype}")
            return cv2.applyColorMap(
                cv2.convertScaleAbs(image, alpha=alpha), cv2.COLORMAP_JET
            )
        elif image.ndim == 3 and channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise NotImplementedError(f"Unknown image shape: {image.shape}")

    def show_images(self, images: List[np.ndarray]):
        """Show the list of images
        :param images: List of np.ndarray images of same shape.
                       Supports depth or RGB color image
                       If depth image, dtype can be np.uint16 or np.floating
                       If RGB image, dtype must be np.uint8
        """
        images = [self.preprocess_image(image) for image in images]
        if len(images) == 0:
            return
        image_shapes = [image.shape for image in images]
        assert len(set(image_shapes)) == 1, \
            f"Not all images are the same shape: {image_shapes}"

        if len(images) == 1:
            vis_image = images[0]
        elif len(images) in [2, 3]:
            vis_image = np.hstack(images)
        elif len(images) == 4:
            vis_image = np.vstack([
                np.hstack(images[:2]), np.hstack(images[2:])
            ])
        elif len(images) == 6:
            vis_image = np.vstack([
                np.hstack(images[:3]), np.hstack(images[3:])
            ])
        elif len(images) == 8:
            vis_image = np.vstack([
                np.hstack(images[:4]), np.hstack(images[4:])
            ])
        elif len(images) == 9:
            vis_image = np.vstack([
                np.hstack(images[:3]), np.hstack(images[3:6]),
                np.hstack(images[6:])
            ])
        else:
            raise NotImplementedError(
                f"Cannot handle plotting {len(images)} images"
            )
        cv2.imshow(self.window_name, vis_image)
        cv2.waitKey(1)

    def clear_image(self):
        """Show a black image"""
        cv2.imshow(self.window_name, np.zeros((128, 128, 3), dtype=np.uint8))
        cv2.waitKey(1)

    def render(self):
        """Update renderer to show image
        and respond to mouse and keyboard events"""
        cv2.waitKey(1)

    def close(self):
        cv2.destroyWindow(self.window_name)

    def __del__(self):
        self.close()
