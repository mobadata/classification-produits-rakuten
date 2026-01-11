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

## Auteurs

- Moussa BA
- Ben djabir Chipinda
- Nedjai Azzedine

