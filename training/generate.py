
import os
import sys
import torch 
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_config import DATA_PATH

from training.dataset import Data
from model.gpt import GPT

    
def generate(model,dataset,start_text, max_new_tokens, strategy='sampling'):
    
    assert max_new_tokens>0 ,'max new token doit etre positif '
    assert len(start_text)>0,'start text vide'
    

    try:
        
      tokens=[dataset.char_to_int[i] for i in start_text]
    except KeyError as e :
        raise ValueError(f"Caractère inconnu dans start_text : {e}")
    tensor_tokens=torch.tensor(tokens,dtype=torch.long)
    x = tensor_tokens.unsqueeze(0) 
    with torch.no_grad():
     for _ in range(max_new_tokens):
        x_crop = x[:, -256:]
        logits = model(x_crop)[:, -1, :]
        probs=F.softmax(logits,dim=-1)
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        x=torch.cat([x,next_token],dim=1)
    tokens_generated = x.squeeze(0).tolist()
    text = ''.join([dataset.int_to_char[t] for t in tokens_generated])
    return text
    
    
def main():
    try: 
     dataset = Data(DATA_PATH)
    except FileNotFoundError:
        raise FileNotFoundError("Dataset introuvable !")  

    model = GPT()
    try:
        model.load_state_dict(torch.load("best_model.pth"))
    except FileNotFoundError:
        raise FileNotFoundError("best_model.pth introuvable !")
    model.eval()
    
    result = generate(model, dataset, "FATHER:", 200)
    print(result)

if __name__ == "__main__":
    main()

    