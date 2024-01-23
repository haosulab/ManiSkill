import sapien
import torch

sapien.set_log_level("info")
sapien.render.set_log_level("info")

sapien.physx.enable_gpu()
sapien.set_cuda_tensor_backend("torch")


def create_scene(offset):
    scene = sapien.Scene()
    scene.physx_system.set_scene_offset(scene, offset)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    return scene


scene0 = create_scene([0, 0, 0])
scene1 = create_scene([10, 0, 0])

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(*[nn.Linear(3, 3), nn.Tanh(), nn.Linear(3, 1)])

    def forward(self, x):
        return self.mlp(x)


device = torch.device("cuda:0")


def loss(x):
    return x**2


model = MLP().to(device)
x = torch.rand(3).to(device)
output = model(x)
loss(output).backward()
