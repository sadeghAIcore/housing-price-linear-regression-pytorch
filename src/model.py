
import torch
from torch import nn

class LinearRegression(nn.Module) :
    def __init__(self , input_dim):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_dim, 32),  
            nn.Linear(32, 1)                
        )


    def forward(self , x):
        return self.layer_stack(x)
