import numpy as np
from pathlib import Path
from mani_skill.utils import common, gym_utils
from mani_skill.utils.wrappers import RecordEpisode
from common.logger import Logger
from mani_skill.utils.visualization.misc import (
    images_to_video,
    tile_images,
)

class RecordEpisodeWrapper(RecordEpisode):
    def __init__(
        self,
        env,
        output_dir,
        save_trajectory=True,
        trajectory_name=None,
        save_video=True,
        info_on_video=False,
        save_on_reset=True,
        save_video_trigger=None,
        max_steps_per_video=None,
        clean_on_close=True,
        record_reward=True,
        video_fps=30,
        source_type=None,
        source_desc=None,
        logger:Logger=None,
    ):
        super().__init__(env, output_dir, save_trajectory, trajectory_name, save_video, info_on_video, save_on_reset, save_video_trigger, max_steps_per_video, clean_on_close, record_reward, video_fps, source_type, source_desc)
        self.logger = logger
        self.untiled_render_images = [] # render_images but not tiled by num_envs. for organized wandb video

    def capture_image(self):
        img = self.env.render()
        img = common.to_numpy(img)
        if len(img.shape) > 3:
            tiled_img = tile_images(img, nrows=self.video_nrows)
        else:
            tiled_img = img
            img = np.expand_dims(img, axis=0)
        self.untiled_render_images.append(img) # (num_envs, h, w, 3)
        return tiled_img # (h, w, 3)

    def flush_video(
        self,
        name=None,
        suffix="",
        verbose=False,
        ignore_empty_transition=True,
        save: bool = True,
    ):
        """
        Flush a video of the recorded episode(s) anb by default saves it to disk

        Arguments:
            name (str): name of the video file. If None, it will be named with the episode id.
            suffix (str): suffix to add to the video file name
            verbose (bool): whether to print out information about the flushed video
            ignore_empty_transition (bool): whether to ignore trajectories that did not have any actions
            save (bool): whether to save the video to disk
        """
        if len(self.render_images) == 0:
            return
        if ignore_empty_transition and len(self.render_images) == 1:
            return
        if save:
            self._video_id += 1
            if name is None:
                video_name = "{}".format(self._video_id)
                if suffix:
                    video_name += "_" + suffix
                if self._avoid_overwriting_video:
                    while (
                        Path(self.output_dir)
                        / (video_name.replace(" ", "_").replace("\n", "_") + ".mp4")
                    ).exists():
                        self._video_id += 1
                        video_name = "{}".format(self._video_id)
                        if suffix:
                            video_name += "_" + suffix
            else:
                video_name = name
            if self.logger.save_video_local:
                images_to_video(
                    self.render_images,
                    str(self.output_dir),
                    video_name=video_name,
                    fps=self.video_fps,
                    verbose=verbose,
                )
            if self.logger is not None:
                untiled_render_images = np.array(self.untiled_render_images)
                untiled_render_images = untiled_render_images.transpose(1, 0, 2, 3, 4)
                self.logger.add_wandb_video(untiled_render_images)
        self._video_steps = 0
        self.render_images = []
        self.untiled_render_images = []