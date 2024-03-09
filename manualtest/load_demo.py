from torch.utils.data import DataLoader

from mani_skill.trajectory.dataset import ManiSkillTrajectoryDataset

if __name__ == "__main__":

    dataset_path = "demos/motionplanning/PickCube-v1/test.h5"

    ds = ManiSkillTrajectoryDataset(dataset_path)
    ds[:10]
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(dl))
    import ipdb

    ipdb.set_trace()
