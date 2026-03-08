#  Mini-GPT — Modèle de Langage from Scratch

> Un modèle GPT miniature construit entièrement from scratch en PyTorch, entraîné sur les œuvres complètes de Shakespeare pour générer du texte à la manière élisabéthaine.

---

##  Description du projet

Ce projet implémente un **modèle de langage de type GPT** (Generative Pre-trained Transformer) from scratch, sans utiliser de bibliothèques de haut niveau comme HuggingFace. L'objectif est de comprendre en profondeur les mécanismes qui font fonctionner les grands modèles de langage modernes (ChatGPT, Claude, Gemini...).

Le modèle apprend à **prédire le prochain caractère** dans une séquence de texte. Après entraînement, il peut générer du texte qui imite le style shakespearien.

---

##  Architecture du projet

```
mini_gpt/
│
├── data/
│   └── input.txt              # Dataset : œuvres complètes de Shakespeare (~1MB)
│
├── model/
│   ├── attention.py           # Multi-head Self-Attention
│   ├── feedforward.py         # Couche Feed-Forward
│   ├── block.py               # Bloc Transformer complet
│   └── gpt.py                 # Modèle GPT assemblé
│
├── training/
│   ├── dataset.py             # Chargement, tokenisation et préparation des données
│   ├── trainer.py             # Boucle d'entraînement avec validation et early stopping
│   └── generate.py            # Génération de texte
│
├── gpt_config.py              # Hyperparamètres centralisés
└── README.md
```

---

## 🏗️ Architecture du modèle

Le modèle suit l'architecture **Transformer Decoder** introduite dans le papier *"Attention is All You Need"* (Vaswani et al., 2017).

```
Entrée (tokens)
      ↓
Token Embedding + Positional Embedding
      ↓
┌─────────────────────────────┐
│     Bloc Transformer × 4    │
│                             │
│  LayerNorm                  │
│  Multi-Head Self-Attention  │
│  + Connexion résiduelle     │
│                             │
│  LayerNorm                  │
│  Feed-Forward Network       │
│  + Connexion résiduelle     │
└─────────────────────────────┘
      ↓
LayerNorm finale
      ↓
Linear → vocab_size
      ↓
Logits (scores par token)
```

### Hyperparamètres

| Paramètre | Valeur | Description |
|---|---|---|
| `vocab_size` | 65 | Nombre de caractères uniques |
| `n_embd` | 128 | Dimension des embeddings |
| `n_head` | 4 | Nombre de têtes d'attention |
| `n_layer` | 4 | Nombre de blocs Transformer |
| `block_size` | 256 | Fenêtre de contexte (tokens) |
| `batch_size` | 32 | Taille du batch |
| `dropout` | 0.2 | Taux de dropout |
| `learning_rate` | 1e-3 | Taux d'apprentissage |
| `max_iters` | 5000 | Nombre d'itérations max |

---

## 🔍 Comment fonctionne l'entraînement d'un LLM ?

### 1. La tokenisation

Le texte brut est converti en **séquence de nombres**. Ici on utilise une tokenisation **par caractère** — chaque caractère unique du vocabulaire reçoit un identifiant numérique :

```
'a' → 53
'b' → 54
' ' → 1
'\n' → 0
...
```

Le vocabulaire de Shakespeare contient **65 caractères uniques**.

### 2. Les embeddings

Chaque token (nombre) est converti en un **vecteur dense** de 128 dimensions. Ces vecteurs sont appris pendant l'entraînement — les tokens similaires finissent par avoir des vecteurs proches.

On utilise deux types d'embeddings :
- **Token embedding** : "quel caractère suis-je ?"
- **Positional embedding** : "à quelle position suis-je ?"

Les deux sont additionnés pour donner au modèle à la fois l'identité et la position de chaque token.

### 3. Le mécanisme d'attention

C'est le cœur du Transformer. Pour chaque token, l'attention calcule **à quel point il doit regarder les autres tokens** pour prédire le suivant.

Pour chaque token on calcule trois vecteurs :
- **Q (Query)** : "qu'est-ce que je cherche ?"
- **K (Key)** : "qu'est-ce que j'offre ?"
- **V (Value)** : "qu'est-ce que je transmets si on me choisit ?"

La formule d'attention :

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √d_k) × V
```

**Exemple avec "Papa mange du riz" :**

```
"riz" regarde :
  44% → "mange"  (il sait qu'il est l'objet d'une action)
  22% → "Papa"   (il sait qui fait l'action)
  17% → "du"     (article)
  17% → lui-même
```

Après l'attention, "riz" n'est plus juste le mot "riz" — son vecteur contient **tout le contexte de la phrase** !

### 4. Le masque causal

Le modèle génère du texte de **gauche à droite**. Pour éviter qu'il "triche" en regardant les tokens futurs pendant l'entraînement, on applique un masque triangulaire :

```
        t0    t1    t2    t3
t0    [ ✅   ❌    ❌    ❌ ]  ← voit seulement lui-même
t1    [ ✅   ✅    ❌    ❌ ]  ← voit t0 et lui-même
t2    [ ✅   ✅    ✅    ❌ ]  ← voit t0, t1 et lui-même
t3    [ ✅   ✅    ✅    ✅ ]  ← voit tout
```

### 5. Le Multi-Head Attention

Au lieu d'une seule attention, on en utilise **4 en parallèle** (une par tête). Chaque tête apprend des relations différentes dans le texte :
- Tête 1 → relations syntaxiques
- Tête 2 → relations sémantiques
- Tête 3 → relations de position
- Tête 4 → autres patterns

Les résultats sont concaténés puis projetés pour recombiner les informations.

### 6. Le Feed-Forward Network

Après l'attention, chaque token passe dans un petit réseau de neurones qui lui permet de "réfléchir" individuellement :

```
Linear(128 → 512) → ReLU → Linear(512 → 128)
```

### 7. Les connexions résiduelles

À chaque sous-couche on ajoute l'entrée à la sortie :

```
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

Cela permet aux gradients de circuler librement pendant la backpropagation — sans ça les réseaux profonds ont du mal à apprendre.

### 8. La tâche d'entraînement

La tâche est simple : **prédire le prochain token**. Pour chaque séquence `x`, on crée une cible `y` décalée d'un token :

```
x = [18, 47, 56, 57]
y = [47, 56, 57, 58]

Le modèle voit 18 → doit prédire 47
Le modèle voit 18, 47 → doit prédire 56
...
```

La loss utilisée est la **cross-entropy** — elle pénalise le modèle quand le bon token n'a pas le score le plus élevé.

---

## 🚀 Installation

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

## 🏋️ Entraîner le modèle

```bash
python training/trainer.py
```

L'entraînement affiche la loss train et validation toutes les 100 itérations :

Le meilleur modèle est sauvegardé automatiquement dans `best_model.pth`.

---

## 📊 Exemples de texte généré

Après 2000 itérations d'entraînement (train loss ≈ 1.82) :

**Stratégie Sampling :**
```
ESCALUS:
What say you to this? I have been a man
That I have seen the sun of the world,
And the world is not the world of the world.
```

> 💡 Le modèle génère du texte grammaticalement cohérent qui ressemble au style shakespearien, même si le sens reste limité — c'est un mini-modèle entraîné sur CPU !

---

##  Courbe d'apprentissage

La courbe d'apprentissage est sauvegardée automatiquement dans `learning_curve.png` après l'entraînement.

---

##  Ce qu'on a appris

En construisant ce projet from scratch on a compris :

- Comment fonctionne la **tokenisation** par caractère
- Le mécanisme de **self-attention** et pourquoi il est si puissant
- L'architecture complète d'un **Transformer Decoder**
- Comment **entraîner** un modèle de langage avec PyTorch
- Le rôle des **connexions résiduelles** et de la **Layer Normalization**

---

## 📚 Références

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017


---

*Projet réalisé pas à pas, en comprenant chaque mécanisme avant de le coder.*
