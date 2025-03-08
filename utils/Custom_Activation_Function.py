# Custom_Activation_Function

import torch
import torch.nn as nn

class ScaledSigmoid(nn.Module):
    def __init__(self, scale=1): # GroundT range ( 0 ~ scale )
        super(ScaledSigmoid, self).__init__()
        self.scale = scale
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.scale * self.sigmoid(x)
    def __repr__(self):
        return (f"ScaledSigmoid(scale={self.scale})")
    
class ScaledTanh(nn.Module):
    def __init__(self, scale=1):
        super(ScaledTanh, self).__init__()
        self.scale = scale
    
    def forward(self, x):
        return self.scale * ((torch.tanh(x) + 1)/2) # range 0~1
    def __repr__(self):
        return (f"ScaledTanh(scale={self.scale})")
    
class ReLU_clamp(nn.Module):
    def __init__(self, max=None):
        super(ReLU_clamp, self).__init__()
        self.clamp_max = max
    def forward(self, x):
        return torch.clamp(torch.relu(x), max= self.clamp_max)
    def __repr__(self):
        return (f"ReLU_clamp(max={self.clamp_max})")