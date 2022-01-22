import pathlib

import torch
import torch.nn.functional as F
from torch.nn import Module, Linear
from torch.utils.tensorboard import SummaryWriter

class Network(Module):
    def __init__(self):
        super().__init__()

        self.fc1 = Linear(10, 20)
        self.fc2 = Linear(20, 30)
        self.fc3 = Linear(30, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.relu(x)

        return x

if __name__ == '__main__':
    log_dir = pathlib.Path.cwd()/ "tensorboard_logs"
    writer = SummaryWriter(log_dir)

    x = torch.rand(1, 10)
    net = Network()

    def activation_hook(inst, inp, out):
        """Runs activation hook

        Parameters:
        ----------------------------
        inst: torch.nn.Module
            The layer that we want to attach the hook to
        
        """