import os
from typing import Dict, List, Optional

import cv2
import imageio
import numpy as np
import tqdm


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    verbose: bool = True,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    References:
        https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/utils.py
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    output_path = os.path.join(output_dir, video_name)
    writer = imageio.get_writer(output_path, fps=fps, quality=quality, **kwargs)
    if verbose:
        print(f"Video created: {output_path}")
        images_iter = tqdm.tqdm(images)
    else:
        images_iter = images
    for im in images_iter:
        writer.append_data(im)
    writer.close()


def normalize_depth(depth, min_depth=0, max_depth=None):
    if min_depth is None:
        min_depth = np.min(depth)
    if max_depth is None:
        max_depth = np.max(depth)
    depth = (depth - min_depth) / (max_depth - min_depth)
    depth = np.clip(depth, 0, 1)
    return depth


def observations_to_images(observations, max_depth=None) -> List[np.ndarray]:
    """Parse images from camera observations."""
    images = []
    for key in observations:
        if "rgb" in key or "Color" in key:
            rgb = observations[key][..., :3]
            if rgb.dtype == np.float32:
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
            images.append(rgb)
        elif "depth" in key or "Position" in key:
            depth = observations[key]
            if "Position" in key:  # [H, W, 4]
                depth = -depth[..., 2:3]
            # [H, W, 1]
            depth = normalize_depth(depth, max_depth=max_depth)
            depth = np.clip(depth * 255, 0, 255).astype(np.uint8)
            depth = np.repeat(depth, 3, axis=-1)
            images.append(depth)
        elif "seg" in key:
            seg: np.ndarray = observations[key]  # [H, W, 1]
            assert seg.ndim == 3 and seg.shape[-1] == 1, seg.shape
            # A heuristic way to colorize labels
            seg = np.uint8(seg * [11, 61, 127])  # [H, W, 3]
            images.append(seg)
        elif "Segmentation" in key:
            seg: np.ndarray = observations[key]  # [H, W, 4]
            assert seg.ndim == 3 and seg.shape[-1] == 4, seg.shape
            # A heuristic way to colorize labels
            visual_seg = np.uint8(seg[..., 0:1] * [11, 61, 127])  # [H, W, 3]
            actor_seg = np.uint8(seg[..., 1:2] * [11, 61, 127])  # [H, W, 3]
            images.append(visual_seg)
            images.append(actor_seg)
    return images


def tile_images(images: List[np.ndarray]) -> np.ndarray:
    """Tile multiple images to a single image. Support non-equal size."""
    # Sort images in descending order of vertical height
    images = sorted(images, key=lambda x: x.shape[0], reverse=True)

    columns = []
    max_h = images[0].shape[0]
    cur_h = 0
    cur_w = images[0].shape[1]

    # Arrange images in columns from left to right
    column = []
    for im in images:
        if cur_h + im.shape[0] <= max_h and cur_w == im.shape[1]:
            column.append(im)
            cur_h += im.shape[0]
        else:
            columns.append(column)
            column = [im]
            cur_h, cur_w = im.shape[0:2]
    columns.append(column)

    # Tile columns
    total_width = sum(x[0].shape[1] for x in columns)
    output_image = np.zeros((max_h, total_width, 3), dtype=images[0].dtype)
    cur_x = 0
    for column in columns:
        cur_w = column[0].shape[1]
        next_x = cur_x + cur_w
        column_image = np.concatenate(column, axis=0)
        cur_h = column_image.shape[0]
        output_image[:cur_h, cur_x:next_x] = column_image
        cur_x = next_x
    return output_image


def put_text_on_image(image: np.ndarray, lines: List[str]):
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()

    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            (0, 255, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def append_text_to_image(image: np.ndarray, lines: List[str]):
    r"""Appends text left to an image of size (height, width, channels).
    The returned image has white text on a black background.
    Args:
        image: the image to put text
        text: a string to display
    Returns:
        A new image with text inserted left to the input image
    See also:
        habitat.utils.visualization.utils
    """
    # h, w, c = image.shape
    font_size = 0.5
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    # text_image = blank_image[0 : y + 10, 0:w]
    # final = np.concatenate((image, text_image), axis=0)
    final = np.concatenate((blank_image, image), axis=1)
    return final


def put_info_on_image(image, info: Dict[str, float], extras=None, overlay=True):
    lines = [f"{k}: {v:.3f}" for k, v in info.items()]
    if extras is not None:
        lines.extend(extras)
    if overlay:
        return put_text_on_image(image, lines)
    else:
        return append_text_to_image(image, lines)
