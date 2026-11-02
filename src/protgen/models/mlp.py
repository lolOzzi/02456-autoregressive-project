import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        
        self.fully_connected = nn.Sequential(
                nn.Linear(3, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 23)
        )
    
    def forward(self, x):
        x = self.fully_connected(x)
        return x
