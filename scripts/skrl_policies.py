import torch
import torch.nn as nn

from torchvision import transforms

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

# Definitions
DEBUG = True

def force_cudnn_initialization():
    s = 32
    dev = torch.device('cuda')
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))


class VisionBasedTraining:
    LINEAR_SIZE = 9216 #230400 (512) and 50176 (256) and 9216 (128)
    class Policy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False,
                    clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )

            self.num_actions = action_space.shape[-1]
            
            self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(VisionBasedTraining.LINEAR_SIZE, 512),
                nn.ReLU(),
                nn.Linear(512, 16),
                nn.Tanh(),
                nn.Linear(16, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, self.num_actions)
            )
            self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        def compute(self, inputs, role):
            # view (samples, width * height * channels) -> (samples, width, height, channels)
            # permute (samples, width, height, channels) -> (samples, channels, width, height)
            x = inputs["states"]["rgb"]
            observation_shape = self.observation_space["rgb"].shape
            x = x.view(-1, *observation_shape[1:])
            x = x.permute(0, 3, 1, 2).float()

            if DEBUG:
                pil_img = transforms.ToPILImage()(x[0])
                pil_img.save("VALUE_TEST_RGB.png")

            x = self.cnn(x) # torch.Size([1, 64, 28, 28]) 28-256, 60-512, 12-128
            x = self.linear(x)
            return 10 * torch.tanh(x), self.log_std_parameter, {}   # JetBotEnv action_space is -10 to 10

    class Value(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.cnn = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU()
                )
            
            self.linear = nn.Sequential(
                nn.Flatten(),
                nn.Linear(VisionBasedTraining.LINEAR_SIZE, 512),
                nn.ReLU(),
                nn.Linear(512, 16),
                nn.Tanh(),
                nn.Linear(16, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )

        def compute(self, inputs, role):
            # view (samples, width * height * channels) -> (samples, width, height, channels)
            # permute (samples, width, height, channels) -> (samples, channels, width, height)
            x = inputs["states"]["rgb"]
            observation_shape = self.observation_space["rgb"].shape
            x = x.view(-1, *observation_shape[1:])
            x = x.permute(0, 3, 1, 2).float()
            x = self.cnn(x)
            x = self.linear(x)
            return x, {}

class StateBasedTraining:
    LINEAR_SIZE = 16
    class Policy(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False,
                    clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
            Model.__init__(self, observation_space, action_space, device)
            GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(StateBasedTraining.LINEAR_SIZE, 512),
                nn.ReLU(),
                nn.Linear(512, 16),
                nn.Tanh(),
                nn.Linear(16, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, self.num_actions)
            )
            self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        def compute(self, inputs, role):
            x = inputs["states"]
            x = self.net(x)
            return 10 * torch.tanh(x), self.log_std_parameter, {}   # JetBotEnv action_space is -10 to 10

    class Value(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device, clip_actions=False):
            Model.__init__(self, observation_space, action_space, device)
            DeterministicMixin.__init__(self, clip_actions)

            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(StateBasedTraining.LINEAR_SIZE, 512),
                nn.ReLU(),
                nn.Linear(512, 16),
                nn.Tanh(),
                nn.Linear(16, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
            )

        def compute(self, inputs, role):
            x = inputs["states"]
            x = self.net(x)
            return x, {}
