import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_dim, width, depth):
        """
        depth: number of hidden layers.
        width: number of neurons per hidden layer.
        """
        super(DNN, self).__init__()
        layers = []
        
        # Input layer -> First Hidden
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())
        
        # Additional Hidden Layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
            
        # Output Layer
        layers.append(nn.Linear(width, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)