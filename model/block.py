from model.attention import MultiHeadAttention
from model.feed_forward import Feedforward
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_config import block_size,batch_size,n_embd ,n_head ,n_layer,dropout,learning_rate ,max_iters 
import torch 
import torch.nn as nn 
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention=MultiHeadAttention(head_size=n_embd // n_head,n_head=n_head)
        self.ffwd=Feedforward()
        self.ln1=nn.LayerNorm(n_embd)
        self.ln2=nn.LayerNorm(n_embd)
        
    def forward(self,x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x