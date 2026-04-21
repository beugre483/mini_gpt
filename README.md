#  Mini-GPT Modèle de Langage from Scratch

> Un modèle GPT miniature construit entièrement from scratch en PyTorch, entraîné sur les œuvres  de Shakespeare pour générer du texte à la manière élisabéthaine.

---

##  Description du projet

Ce projet implémente un **modèle de langage de type GPT** (Generative Pre-trained Transformer) from scratch, sans utiliser de bibliothèques de haut niveau comme HuggingFace. L'objectif est de comprendre en profondeur les mécanismes qui font fonctionner les grands modèles de langage modernes (ChatGPT, Claude, Gemini...).

Le modèle apprend à **prédire le prochain caractère** dans une séquence de texte. Après entraînement, il peut générer du texte qui imite le style shakespearien.


## Installation

### Prérequis

- Python 3.10+
- Poetry

### Installation

```bash
# Cloner le repo
git clone https://github.com/ton-username/mini-gpt.git
cd mini-gpt/mini_gpt

# Installer les dépendances
poetry install

# Activer l'environnement
poetry shell
```

### Télécharger le dataset

```bash
# Windows PowerShell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" -OutFile "data/input.txt" -UseBasicParsing
```

---

##  Entraîner le modèle

```bash
python training/trainer.py
```

---

**Stratégie Sampling :**
```
ESCALUS:
What say you to this? I have been a man
That I have seen the sun of the world,
And the world is not the world of the world.
```

>  Le modèle génère du texte grammaticalement cohérent qui ressemble au style shakespearien, même si le sens reste limité c'est un mini-modèle entraîné sur CPU !

---


##  Ce qu'on a appris

En construisant ce projet from scratch on a compris :

- Comment fonctionne la **tokenisation** par caractère
- Le mécanisme de **self-attention** et pourquoi il est si puissant
- L'architecture complète d'un **Transformer Decoder**
- Comment **entraîner** un modèle de langage avec PyTorch
- Le rôle des **connexions résiduelles** et de la **Layer Normalization**

---

## Références

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017


---

*Projet réalisé pas à pas, en comprenant chaque mécanisme avant de le coder.*
