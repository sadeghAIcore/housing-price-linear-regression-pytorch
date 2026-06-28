
import torch
from torch import nn

class LinearRegression(nn.Module) :
    def __init__(self , input_dim):
        super.__init__()

        self.Linear = nn.Linear(in_features=input_dim , out_features=1)

    def forward(self , x):
        return self.Linear(x)
