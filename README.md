# Classification de produits e-commerce Rakuten

Projet de classification automatique de produits e-commerce pour Rakuten.

## Objectif

Prédire la catégorie de produits (`prdtypecode`) à partir des informations textuelles (designation, description) et potentiellement des images associées aux produits.

## Structure des données

- `X_train_update.csv` : 84,920 produits d'entraînement avec leurs caractéristiques
- `Y_train_CVw08PX.csv` : 84,917 labels (codes de type de produit)
- `X_test_update.csv` : 13,813 produits de test à classifier
- `Classification de produits e-commerce Rakuten.pdf` : Description détaillée du challenge

### Colonnes des données

**X_train / X_test :**
- `designation` : Nom/titre du produit
- `description` : Description détaillée du produit (peut être vide)
- `productid` : Identifiant unique du produit
- `imageid` : Identifiant de l'image associée

**Y_train :**
- `prdtypecode` : Code de catégorie de produit (variable cible)

## Approche

1. **Exploration des données** : Analyse des distributions, des catégories, de la qualité des textes
2. **Feature Engineering** : Extraction de caractéristiques textuelles (TF-IDF, embeddings, etc.)
3. **Modélisation** : Développement de modèles de classification multi-classes
4. **Évaluation** : Métriques de performance adaptées à la classification multi-classes

## Récupérer le modèle CamemBERT entraîné sur Colab

Si vous avez entraîné CamemBERT sur Google Colab, voici comment récupérer le modèle et l’utiliser dans ce projet.

### 1. Où se trouve le modèle sur Colab ?

- **Si vous avez sauvegardé sur Google Drive** (par ex. `OUTPUT_DIR = '/content/drive/MyDrive/Projet_Rakuten'`)  
  → Ouvrez Google Drive, allez dans le dossier (ex. `Projet_Rakuten`). Vous devriez voir :
  - `models/camembert_finetuned/` (dossier avec `config.json`, `pytorch_model.bin`, `tokenizer_config.json`, etc.)
  - `models/camembert_label_to_id.pkl` et `models/camembert_id_to_label.pkl`
  - `output/predictions_camembert.csv`

- **Si tout est resté dans `/content`** (sans Drive)  
  → Avant de fermer la session Colab, récupérez les fichiers (voir ci‑dessous).

### 2. Télécharger depuis Colab vers votre ordinateur

**Option A – Depuis Google Drive**

1. Sur [drive.google.com](https://drive.google.com), ouvrez le dossier où Colab a enregistré (ex. `Projet_Rakuten`).
2. Clic droit sur le dossier `models` → **Télécharger** (ou sur `camembert_finetuned` si vous ne voulez que le modèle).
3. Si vous voulez les prédictions : téléchargez aussi le dossier ou le fichier `output`.

**Option B – Depuis une session Colab (fichiers dans `/content`)**

Dans une cellule Colab, exécuter pour créer une archive et la télécharger :

```python
# À lancer sur Colab avant de fermer la session
import shutil
import os

os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)
# Si vos fichiers sont dans /content/models et /content/output :
shutil.make_archive('camembert_colab', 'zip', '/content', 'models')
# Puis : Fichier > Télécharger > camembert_colab.zip
```

Ou télécharger le dossier `camembert_finetuned` seul :

```python
shutil.make_archive('camembert_finetuned', 'zip', '/content/models', 'camembert_finetuned')
# Fichier > Télécharger > camembert_finetuned.zip
```

### 3. Où placer les fichiers dans ce projet (sur votre Mac)

À la **racine du projet** (même niveau que `README.md`, `X_train_update.csv`, etc.) :

```
Projet Rakuten/
├── models/
│   ├── camembert_finetuned/     ← dossier complet (config.json, pytorch_model.bin, etc.)
│   ├── camembert_label_to_id.pkl
│   └── camembert_id_to_label.pkl
├── output/
│   └── predictions_camembert.csv   (optionnel)
├── X_train_update.csv
├── ...
```

- Dézippez `camembert_finetuned.zip` (si besoin) et placez le **dossier** `camembert_finetuned` dans `models/`.
- Copiez les deux fichiers `.pkl` dans `models/`.
- Si vous avez téléchargé les prédictions, mettez `predictions_camembert.csv` dans `output/`.

### 4. Utiliser le modèle en local

Ouvrez le notebook `07_modelisation_text_CamemBERT.ipynb` **en local**.  
La cellule « Configuration Google Colab » définit `DATA_DIR = '.'` et `OUTPUT_DIR = '.'`.  
La cellule qui charge le modèle détecte la présence de `models/camembert_finetuned` : si le dossier est là, le notebook charge ce modèle et **ne refait pas l’entraînement**. Vous pouvez enchaîner directement sur l’évaluation et les prédictions.

---

## Auteurs

- Moussa BA
- Ben djabir Chipinda
- Nedjai Azzedine

