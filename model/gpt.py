import torch
import torch.nn as nn
from gpt_config import vocab_size, n_embd, block_size, n_layer, n_head
from model.block import Block

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)  
        self.pos_emb = nn.Embedding(block_size, n_embd)    
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])  
        self.ln_f = nn.LayerNorm(n_embd)                  
        self.head = nn.Linear(n_embd, vocab_size)     
        
    def forward(self,x):
        tok = self.token_emb(x)  # (B, T, n_embd)
        pos = torch.arange(x.shape[1])  # [0, 1, 2, ..., T-1]
        pos = self.pos_emb(pos)          # (T, n_embd)
        x = tok + pos  # (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits