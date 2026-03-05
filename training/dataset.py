import torch
from config import block_size, batch_size
class Data():
   def __init__(self,path_data):
       self.path_data=path_data
       self.block_size=block_size
       self.batch_size=batch_size
       self.tokenize_data()
       self.encode_data()
       self.convert_to_tensor()
       self.split(0.9)
 
   def tokenize_data(self):
        with open(self.path_data,'r') as f:
            self.text=f.read()
        char=sorted(set(self.text))
        self.char_to_int={val:i for i,val in enumerate(char)}
        self.int_to_char={i:val for i,val in enumerate(char)}
        self.vocab_size = len(char)
        
   def encode_data(self):
       self.tokens=[self.char_to_int[i] for i in self.text]
       
   def convert_to_tensor(self):
       self.data=torch.tensor(self.tokens,dtype=torch.long)
        
   def split(self,n):
        size=int(n*len(self.data))
        self.train_data=self.data[:size]
        self.test_data=self.data[size:]
        
   def get_batch(self,split='train'):
        data=self.train_data if split=='train' else self.test_data
        self.indices=torch.randint(len(data)-self.block_size,(self.batch_size,))
        self.x=torch.stack([data[i:i+self.block_size] for i in self.indices])
        self.y=torch.stack([data[i+1:i+self.block_size+1] for i in self.indices])
        return self.x,self.y

    
       
       
        