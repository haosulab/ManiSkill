"""
PushCube-v1 is a basic/common task which defaults to using the panda robot. It is also used as a testing task to check whether a robot with manipulation
capabilities can be simulated and trained properly. The configs below set the pick cube task differently to ensure the cube is within reach of the robot tested
and the camera angles are reasonable.
"""
PUSH_CUBE_CONFIGS = {
    "panda": {
        "goal_radius": 0.1,
        "goal_distance": 0.1,
        "cube_half_size": 0.02,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "sensor_cam_eye_pos": [0.3, 0, 0.6],  # sensor cam is the camera used for visual observation generation
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],  # human cam is the camera used for human rendering (i.e. eval videos)
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "fetch": {
        "goal_radius": 0.1,
        "goal_distance": 0.1,
        "cube_half_size": 0.02,
        "cube_spawn_half_size": 0.1,
        "cube_spawn_center": (0, 0),
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
    "so100": {
        "goal_radius": 0.05,
        "goal_distance": 0.025,
        "cube_half_size": 0.0125,
        "cube_spawn_half_size": 0.03,
        "cube_spawn_center": (-0.5, 0),
        "sensor_cam_eye_pos": [-0.27, 0, 0.4],
        "sensor_cam_target_pos": [-0.56, 0, -0.25],
        "human_cam_eye_pos": [-0.1, 0.3, 0.4],
        "human_cam_target_pos": [-0.46, 0.0, 0.1],
    },
    "xarm6_robotiq": {
        "goal_radius": 0.1,
        "goal_distance": 0.075,
        "cube_half_size": 0.02,
        "cube_spawn_half_size": 0.075,
        "cube_spawn_center": (-0.05, 0),
        "sensor_cam_eye_pos": [0.3, 0, 0.6],
        "sensor_cam_target_pos": [-0.1, 0, 0.1],
        "human_cam_eye_pos": [0.6, 0.7, 0.6],
        "human_cam_target_pos": [0.0, 0.0, 0.35],
    },
}
