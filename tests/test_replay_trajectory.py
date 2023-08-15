import pytest

from tests.utils import CONTROL_MODES_STATIONARY_SINGLE_ARM, ENV_IDS, OBS_MODES


@pytest.mark.parametrize(
    "control_mode",
    [
        "pd_joint_delta_pos",
        "pd_joint_target_delta_pos",
        "pd_joint_vel",
        "pd_ee_delta_pose",
    ],
)
def test_replay_trajectory(control_mode):
    env_id = "PickCube-v0"
    # from mani_skill2.utils.download_demo import main as download_demo, parse_args as download_demo_parse_args
    # download_demo(download_demo_parse_args(args=[env_id]))
    from mani_skill2.trajectory.replay_trajectory import main, parse_args

    main(
        parse_args(
            args=[
                "--traj-path",
                f"demos/rigid_body/{env_id}/trajectory.h5",
                "--save-traj",
                "--target-control-mode",
                control_mode,
                "--count",
                "4",
            ]
        )
    )
