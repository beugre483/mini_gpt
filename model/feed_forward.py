from config import block_size, n_embd
import torch
import torch.nn as nn
import torch.nn.functional as F

class Feedforward(nn.Module):
    def __init__(self):
        super().__init__()
        self.couche1=nn.Linear(n_embd,4*n_embd)
        self.couche2=nn.Linear(4*n_embd,n_embd)
        
    def forward(self,x):
        x=self.couche1(x)
        x=F.relu(x)
        x=self.couche2(x)
        return x
        
        
        