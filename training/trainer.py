import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpt_config import learning_rate, max_iters, block_size, batch_size,DATA_PATH
from model.gpt import GPT
from training.dataset import Data

def evaluate(model, dataset, eval_batches=20):
    model.eval()  # désactive le dropout pendant l'évaluation
    total_loss = 0
    
    with torch.no_grad():  
        for _ in range(eval_batches):
            x, y = dataset.get_batch('val')
            logits = model(x)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            y = y.view(B*T)
            loss = nn.functional.cross_entropy(logits, y)
            total_loss += loss.item()
    
    model.train()  # réactive le dropout pour continuer l'entraînement
    return total_loss / eval_batches

def train():
    dataset = Data(DATA_PATH)
    model = GPT()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Pour la courbe d'apprentissage
    train_losses = []
    val_losses = []
    iters = []
    
    # Pour l'early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    for i in range(max_iters):
        # Batch d'entraînement
        x, y = dataset.get_batch('train')
        logits = model(x)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        y = y.view(B*T)
        loss = nn.functional.cross_entropy(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Évaluation toutes les 100 itérations
        if i % 100 == 0:
            val_loss = evaluate(model, dataset)
            train_losses.append(loss.item())
            val_losses.append(val_loss)
            iters.append(i)
            
            print(f"étape {i}/{max_iters} | train loss: {loss.item():.4f} | val loss: {val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pth")  # sauvegarde le meilleur modèle
                print(f"   Meilleur modèle sauvegardé ! val_loss = {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"   Pas d'amélioration ({patience_counter}/{patience})")
                
                if patience_counter >= patience:
                    print("  Early stopping !")
                    break
    
    # Courbe d'apprentissage
    plt.figure(figsize=(10, 5))
    plt.plot(iters, train_losses, label='Train loss')
    plt.plot(iters, val_losses, label='Val loss')
    plt.xlabel('Itérations')
    plt.ylabel('Loss')
    plt.title('Courbe d\'apprentissage')
    plt.legend()
    plt.savefig('learning_curve.png')
    plt.show()
    print("Courbe sauvegardée dans learning_curve.png")

if __name__ == "__main__":
    train()