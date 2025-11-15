from copy import deepcopy

import numpy as np
import sapien
import torch

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.robocasa.objects.kitchen_object_utils import (
    sample_kitchen_object,
)
from mani_skill.utils.scene_builder.robocasa.objects.objects import MJCFObject
from mani_skill.utils.scene_builder.robocasa.scene_builder import RoboCasaSceneBuilder
from mani_skill.utils.scene_builder.robocasa.utils import scene_registry
from mani_skill.utils.scene_builder.robocasa.utils.placement_samplers import (
    RandomizationError,
)
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env(
    "RoboCasaKitchen-v1", max_episode_steps=100, asset_download_ids=["RoboCasa"]
)
class RoboCasaKitchenEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["fetch", "none"]
    SUPPORTED_REWARD_MODES = ["none"]
    """
    Initialized a Base Kitchen environment.

    Args:
        robots: Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)

        env_configuration (str): Specifies how to position the robot(s) within the environment. Default is "default",
            which should be interpreted accordingly by any subclasses.

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        base_types (None or str or list of str): type of base, used to instantiate base models from base factory.
            Default is "default", which is the default base associated with the robot(s) the 'robots' specification.
            None results in no base, and any other (valid) model overrides the default base. Should either be
            single str if same base type is to be used for all robots or else it should be a list of the same
            length as "robots" param

        gripper_types (None or str or list of str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default grippers(s) associated
            with the robot(s) the 'robots' specification. None removes the gripper, and any other (valid) model
            overrides the default gripper. Should either be single str if same gripper type is to be used for all
            robots or else it should be a list of the same length as "robots" param

        initialization_noise (dict or list of dict): dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        placement_initializer (ObjectPositionSampler): if provided, will be used to place objects on every reset,
            else a UniformRandomSampler is used by default.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        render_gpu_device_id (int): corresponds to the GPU device id to use for offscreen rendering.
            Defaults to -1, in which case the device will be inferred from environment variables
            (GPUS or CUDA_VISIBLE_DEVICES).

        control_freq (float): how many control signals to receive in every second. This sets the abase of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).


        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

        renderer (str): Specifies which renderer to use.

        renderer_config (dict): dictionary for the renderer configurations

        init_robot_base_pos (str): name of the fixture to place the near. If None, will randomly select a fixture.

        seed (int): environment seed. Default is None, where environment is unseeded, ie. random

        layout_and_style_ids (list of list of int): list of layout and style ids to use for the kitchen.

        layout_ids ((list of) LayoutType or int):  layout id(s) to use for the kitchen. -1 and None specify all layouts
            -2 specifies layouts not involving islands/wall stacks, -3 specifies layouts involving islands/wall stacks,
            -4 specifies layouts with dining areas.

        style_ids ((list of) StyleType or int): style id(s) to use for the kitchen. -1 and None specify all styles.

        generative_textures (str): if set to "100p", will use AI generated textures

        obj_registries (tuple of str): tuple containing the object registries to use for sampling objects.
            can contain "objaverse" and/or "aigen" to sample objects from objaverse, AI generated, or both.

        obj_instance_split (str): string for specifying a custom set of object instances to use. "A" specifies
            all but the last 3 object instances (or the first half - whichever is larger), "B" specifies the
            rest, and None specifies all.

        use_distractors (bool): if True, will add distractor objects to the scene

        translucent_robot (bool): if True, will make the robot appear translucent during rendering

        randomize_cameras (bool): if True, will add gaussian noise to the position and rotation of the
            wrist and agentview cameras
    """

    EXCLUDE_LAYOUTS = []

    def __init__(
        self,
        *args,
        robot_uids="fetch",
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        use_camera_obs=True,
        use_object_obs=True,  # currently unused variable
        reward_scale=1.0,  # currently unused variable
        reward_shaping=False,  # currently unused variables
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="robot0_agentview_center",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        # hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        renderer="mujoco",
        renderer_config=None,
        init_robot_base_pos=None,
        seed=None,
        layout_and_style_ids=None,
        layout_ids=None,
        style_ids=None,
        scene_split=None,  # unsued, for backwards compatibility
        generative_textures=None,
        obj_registries=("objaverse",),
        obj_instance_split=None,
        use_distractors=False,
        translucent_robot=False,
        randomize_cameras=False,
        fixtures_only=False,
        **kwargs,
    ):
        self.init_robot_base_pos = init_robot_base_pos

        # object placement initializer
        self.placement_initializer = placement_initializer
        self.obj_registries = obj_registries
        self.obj_instance_split = obj_instance_split

        if layout_and_style_ids is not None:
            assert (
                layout_ids is None and style_ids is None
            ), "layout_ids and style_ids must both be set to None if layout_and_style_ids is set"
            self.layout_and_style_ids = layout_and_style_ids
        else:
            layout_ids = scene_registry.unpack_layout_ids(layout_ids)
            style_ids = scene_registry.unpack_style_ids(style_ids)
            self.layout_and_style_ids = [(l, s) for l in layout_ids for s in style_ids]

        # remove excluded layouts
        self.layout_and_style_ids = [
            (int(l), int(s))
            for (l, s) in self.layout_and_style_ids
            if l not in self.EXCLUDE_LAYOUTS
        ]

        assert generative_textures in [None, False, "100p"]
        self.generative_textures = generative_textures

        self.use_distractors = use_distractors
        self.translucent_robot = translucent_robot
        self.randomize_cameras = randomize_cameras

        # intialize cameras
        # self._cam_configs = deepcopy(CamUtils.CAM_CONFIGS)

        self.use_object_obs = use_object_obs
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        ### code from original robocasa env class ###
        self._ep_meta = {}
        self.fixtures_only = fixtures_only
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(spacing=8, control_freq=20)

    @property
    def _default_sensor_configs(self):
        # TODO (fix cameras to be where robocasa places them)
        pose = sapien_utils.look_at([3.0, -7.5, 2.5], [3.0, 0.0, 1.0])
        return [
            CameraConfig("base_camera", pose, 128, 128, 60 * np.pi / 180, 0.01, 100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([3.0, -7.5, 2.5], [3.0, 0.0, 1.0])
        return CameraConfig(
            "render_camera", pose, 2048, 2048, 60 * np.pi / 180, 0.01, 100
        )

    @property
    def _default_viewer_camera_config(self):
        return CameraConfig(
            uid="viewer",
            pose=sapien.Pose([0, 0, 1]),
            width=1920,
            height=1080,
            shader_pack="default",
            near=0.0,
            far=1000,
            fov=60 * np.pi / 180,
        )

    def _load_agent(self, options: dict):
        ps = torch.zeros((self.num_envs, 3), device=self.device)
        ps[:, 2] = 5
        super()._load_agent(options, Pose.create_from_pq(p=ps))

    def _load_scene(self, options: dict):
        self.scene_builder = RoboCasaSceneBuilder(self)
        self.scene_builder.build()
        # self.fixtures = data["fixtures"]
        # self.actors = data["actors"]
        # self.fixture_configs = data["fixture_configs"]
        self.fixture_refs = []
        self.objects = []
        self.object_cfgs = []
        self.object_actors = []
        for _ in range(self.num_envs):
            self.fixture_refs.append({})
            self.objects.append({})
            self.object_cfgs.append({})
            self.object_actors.append({})

        # hacky way to ensure robocasa task classes can be easily imported into maniskill
        # by manually setting into self the current scene idx to be loaded and checking if _get_obj_cfgs exists
        if not self.fixtures_only and hasattr(self, "_get_obj_cfgs"):
            for scene_idx in range(self.num_envs):
                self._scene_idx_to_be_loaded = scene_idx
                self._setup_kitchen_references()

                # add objects
                def _create_obj(cfg):
                    if "info" in cfg:
                        """
                        if cfg has "info" key in it, that means it is storing meta data already
                        that indicates which object we should be using.
                        set the obj_groups to this path to do deterministic playback
                        """
                        mjcf_path = cfg["info"]["mjcf_path"]
                        # replace with correct base path
                        new_base_path = os.path.join(
                            robocasa.models.assets_root, "objects"
                        )
                        new_path = os.path.join(
                            new_base_path, mjcf_path.split("/objects/")[-1]
                        )
                        obj_groups = new_path
                        exclude_obj_groups = None
                    else:
                        obj_groups = cfg.get("obj_groups", "all")
                        exclude_obj_groups = cfg.get("exclude_obj_groups", None)
                    object_kwargs, object_info = self.sample_object(
                        obj_groups,
                        exclude_groups=exclude_obj_groups,
                        graspable=cfg.get("graspable", None),
                        washable=cfg.get("washable", None),
                        microwavable=cfg.get("microwavable", None),
                        cookable=cfg.get("cookable", None),
                        freezable=cfg.get("freezable", None),
                        max_size=cfg.get("max_size", (None, None, None)),
                        object_scale=cfg.get("object_scale", None),
                        rng=self._batched_episode_rng[scene_idx],
                    )
                    if "name" not in cfg:
                        cfg["name"] = "obj_{}".format(obj_num + 1)
                    info = object_info
                    object = MJCFObject(self.scene, name=cfg["name"], **object_kwargs)
                    return object, info

                for _ in range(10):
                    objects = {}
                    if "object_cfgs" in self._ep_meta:
                        object_cfgs = self._ep_meta["object_cfgs"]
                        for obj_num, cfg in enumerate(object_cfgs):
                            model, info = _create_obj(cfg)
                            cfg["info"] = info
                            objects[model.name] = model
                            # self.model.merge_objects([model])
                    else:
                        object_cfgs = self._get_obj_cfgs()
                        addl_obj_cfgs = []
                        for obj_num, cfg in enumerate(object_cfgs):
                            cfg["type"] = "object"
                            model, info = _create_obj(cfg)
                            cfg["info"] = info
                            objects[model.name] = model
                            # self.model.merge_objects([model])

                            try_to_place_in = cfg["placement"].get(
                                "try_to_place_in", None
                            )

                            # place object in a container and add container as an object to the scene
                            if try_to_place_in and (
                                "in_container"
                                in cfg["info"]["groups_containing_sampled_obj"]
                            ):
                                container_cfg = {}
                                container_cfg["name"] = cfg["name"] + "_container"
                                container_cfg["obj_groups"] = try_to_place_in
                                container_cfg["placement"] = deepcopy(cfg["placement"])
                                container_cfg["type"] = "object"

                                container_kwargs = cfg["placement"].get(
                                    "container_kwargs", None
                                )
                                if container_kwargs is not None:
                                    for k, v in container_kwargs.values():
                                        container_cfg[k] = v

                                # add in the new object to the model
                                addl_obj_cfgs.append(container_cfg)
                                model, info = _create_obj(container_cfg)
                                container_cfg["info"] = info
                                objects[model.name] = model
                                # self.model.merge_objects([model])

                                # modify object config to lie inside of container
                                cfg["placement"] = dict(
                                    size=(0.01, 0.01),
                                    ensure_object_boundary_in_range=False,
                                    sample_args=dict(
                                        reference=container_cfg["name"],
                                    ),
                                )

                        # prepend the new object configs in
                        object_cfgs = addl_obj_cfgs + object_cfgs

                        # # remove objects that didn't get created
                        # self.object_cfgs = [cfg for cfg in self.object_cfgs if "model" in cfg]
                    self.object_cfgs[scene_idx] = object_cfgs
                    self.objects[scene_idx] = objects
                    placement_initializer = (
                        self.scene_builder._get_placement_initializer(
                            self.scene_builder.scene_data[self._scene_idx_to_be_loaded][
                                "fixtures"
                            ],
                            objects,
                            object_cfgs,
                            rng=self._batched_episode_rng[scene_idx],
                        )
                    )

                    object_placements = None
                    for i in range(10):
                        try:
                            object_placements = placement_initializer.sample(
                                placed_objects=self.scene_builder.scene_data[
                                    self._scene_idx_to_be_loaded
                                ]["fxtr_placements"]
                            )
                        except RandomizationError:
                            #     if macros.VERBOSE:
                            #         print("Ranomization error in initial placement. Try #{}".format(i))
                            continue

                        break
                    if object_placements is None:
                        print("Could not place objects. Trying again with new objects")
                        continue
                    for obj_pos, obj_quat, obj in object_placements.values():
                        obj.pos = obj_pos
                        obj.quat = obj_quat
                        actor = obj.build(scene_idxs=[scene_idx]).actor
                        self.object_actors[scene_idx][obj.name] = {
                            "actor": actor,
                            "pose": sapien.Pose(obj_pos, obj_quat),
                        }
                    break

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            self.scene_builder.initialize(env_idx)
            # TODO (stao): no way to make this faster atm on GPU without a lot of work
            for scene_idx in range(self.num_envs):
                for actor_data in self.object_actors[scene_idx].values():
                    actor_data["actor"].set_pose(actor_data["pose"])

    def evaluate(self):
        return {}

    def _get_obs_extra(self, info: dict):
        return dict()

    """
    Code below is for leveraging the RoboCasa objects dataset and sampling it
    """

    def _setup_kitchen_references(self):
        """
        setup fixtures (and their references). this function is called within load_model function for kitchens
        """
        serialized_refs = self._ep_meta.get("fixture_refs", {})
        # unserialize refs
        self.fixture_refs[self._scene_idx_to_be_loaded] = {
            k: self.scene_builder.get_fixture(
                self.scene_builder.scene_data[self._scene_idx_to_be_loaded]["fixtures"],
                v,
            )
            for (k, v) in serialized_refs.items()
        }

    def register_fixture_ref(self, ref_name, fn_kwargs):
        """
        Registers a fixture reference for later use. Initializes the fixture
        if it has not been initialized yet.

        Args:
            ref_name (str): name of the reference

            fn_kwargs (dict): keyword arguments to pass to get_fixture

        Returns:
            Fixture: fixture object
        """
        if ref_name not in self.fixture_refs:
            scene_idx = self._scene_idx_to_be_loaded
            self.fixture_refs[scene_idx][ref_name] = self.scene_builder.get_fixture(
                self.scene_builder.scene_data[scene_idx]["fixtures"], **fn_kwargs
            )
        return self.fixture_refs[scene_idx][ref_name]

    def sample_object(
        self,
        groups,
        exclude_groups=None,
        graspable=None,
        microwavable=None,
        washable=None,
        cookable=None,
        freezable=None,
        split=None,
        obj_registries=None,
        max_size=(None, None, None),
        object_scale=None,
        rng=None,
    ):
        """
        Sample a kitchen object from the specified groups and within max_size bounds.

        Args:
            groups (list or str): groups to sample from or the exact xml path of the object to spawn

            exclude_groups (str or list): groups to exclude

            graspable (bool): whether the sampled object must be graspable

            washable (bool): whether the sampled object must be washable

            microwavable (bool): whether the sampled object must be microwavable

            cookable (bool): whether whether the sampled object must be cookable

            freezable (bool): whether whether the sampled object must be freezable

            split (str): split to sample from. Split "A" specifies all but the last 3 object instances
                        (or the first half - whichever is larger), "B" specifies the  rest, and None
                        specifies all.

            obj_registries (tuple): registries to sample from

            max_size (tuple): max size of the object. If the sampled object is not within bounds of max size,
                            function will resample

            object_scale (float): scale of the object. If set will multiply the scale of the sampled object by this value


        Returns:
            dict: kwargs to apply to the MJCF model for the sampled object

            dict: info about the sampled object - the path of the mjcf, groups which the object's category belongs to,
            the category of the object the sampling split the object came from, and the groups the object was sampled from
        """
        return sample_kitchen_object(
            groups,
            exclude_groups=exclude_groups,
            graspable=graspable,
            washable=washable,
            microwavable=microwavable,
            cookable=cookable,
            freezable=freezable,
            rng=rng,
            obj_registries=(obj_registries or self.obj_registries),
            split=(split or self.obj_instance_split),
            max_size=max_size,
            object_scale=object_scale,
        )
