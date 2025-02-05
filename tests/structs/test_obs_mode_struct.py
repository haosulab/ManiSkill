from mani_skill.envs.utils.observations import parse_obs_mode_to_struct


def test_state():
    struct = parse_obs_mode_to_struct("state")
    assert struct.state
    assert not struct.state_dict
    assert not struct.visual.rgb
    assert not struct.visual.depth
    assert not struct.visual.segmentation
    assert not struct.visual.position
    assert not struct.visual.normal
    assert not struct.visual.albedo


def test_state_dict():
    struct = parse_obs_mode_to_struct("state_dict")
    assert not struct.state
    assert struct.state_dict
    assert not struct.visual.rgb
    assert not struct.visual.depth
    assert not struct.visual.segmentation
    assert not struct.visual.position
    assert not struct.visual.normal
    assert not struct.visual.albedo


def test_rgbd():
    struct = parse_obs_mode_to_struct("rgbd")
    assert not struct.state
    assert not struct.state_dict
    assert struct.visual.rgb
    assert struct.visual.depth
    assert not struct.visual.segmentation
    assert not struct.visual.position
    assert not struct.visual.normal
    assert not struct.visual.albedo


def test_pointcloud():
    struct = parse_obs_mode_to_struct("pointcloud")
    assert not struct.state
    assert not struct.state_dict
    assert struct.visual.rgb
    assert not struct.visual.depth
    assert struct.visual.segmentation
    assert struct.visual.position
    assert not struct.visual.normal
    assert not struct.visual.albedo


def test_sensor_data():
    struct = parse_obs_mode_to_struct("sensor_data")
    assert not struct.state
    assert not struct.state_dict
    assert struct.visual.rgb
    assert struct.visual.depth
    assert struct.visual.segmentation
    assert struct.visual.position
    assert not struct.visual.normal
    assert not struct.visual.albedo


def test_none():
    struct = parse_obs_mode_to_struct("none")
    assert not struct.state
    assert not struct.state_dict
    assert not struct.visual.rgb
    assert not struct.visual.depth
    assert not struct.visual.segmentation
    assert not struct.visual.position
    assert not struct.visual.normal
    assert not struct.visual.albedo


def test_state_rgb():
    struct = parse_obs_mode_to_struct("state+rgb")
    assert struct.state
    assert not struct.state_dict
    assert struct.visual.rgb
    assert not struct.visual.depth
    assert not struct.visual.segmentation
    assert not struct.visual.position
    assert not struct.visual.normal
    assert not struct.visual.albedo


def test_rgb_depth_state_dict():
    struct = parse_obs_mode_to_struct("rgb+depth+state_dict")
    assert not struct.state
    assert struct.state_dict
    assert struct.visual.rgb
    assert struct.visual.depth
    assert not struct.visual.segmentation
    assert not struct.visual.position
    assert not struct.visual.normal
    assert not struct.visual.albedo


def test_state_dict_depth_pointcloud():
    struct = parse_obs_mode_to_struct("state_dict+depth+pointcloud")
    assert not struct.state
    assert struct.state_dict
    assert struct.visual.rgb
    assert struct.visual.depth
    assert struct.visual.segmentation
    assert struct.visual.position
    assert not struct.visual.normal
    assert not struct.visual.albedo
