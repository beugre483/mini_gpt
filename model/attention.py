import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_config import block_size, n_embd

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.head_size= head_size
        self.query=nn.Linear(n_embd,self.head_size,bias=False)
        self.key=nn.Linear(n_embd,self.head_size,bias=False)
        self.value=nn.Linear(n_embd,self.head_size,bias=False)
        
    def forward(self,x):
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)
        self.score=(q @ k.transpose(-2,-1))/self.head_size**0.5
        mask=torch.tril(torch.ones(self.score.shape[-2],self.score.shape[-1]))
        self.score=self.score.masked_fill(mask==0,float('-inf'))
        self.score = torch.softmax(self.score, dim=-1)
        return self.score @ v
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self,head_size,n_head):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj=nn.Linear(n_embd,n_embd)
        
    def forward(self,x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out=self.proj(out)
        return out 
    