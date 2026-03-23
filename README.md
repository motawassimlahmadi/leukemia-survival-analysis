# Leukemia Survival Analysis

Framework de machine learning pour l'analyse de survie sur des données moléculaires de leucémie. Le pipeline intègre plusieurs modèles de survie, une sélection de features statistique, et une évaluation rigoureuse sur un horizon de 7 ans.

---

## Objectif

Prédire la survie globale de patients atteints de leucémie (OS_YEARS, OS_STATUS) à partir de données moléculaires et cliniques, en utilisant des modèles de survie adaptés aux données censurées.

---

## Architecture du projet

```
leukemia-survival-analysis/
├── main.py                                      # Pipeline principal (CLI)
├── run_all_methods.sh                           # Lance les 3 modèles en séquence
├── run_all_optimized_methods.sh                 # Version optimisée (low_memory)
├── install_dependencies.sh                      # Installation automatique
├── requirements.txt
├── data/
│   └── raw/X_train/molecular_train.csv         # Données moléculaires d'entraînement
├── notebooks/                                   # Analyses exploratoires
├── reports/                                     # Résultats générés
├── src/data/                                    # Prétraitement des données
└── Annexe_projets_techniques_Lahmadi_Motawassim.pdf  # Documentation technique
```

---

## Pipeline ML

```
Données moléculaires + Données cliniques
          │
          ▼
     Jointure sur ID
          │
          ▼
  Suppression colonnes catégorielles / identifiants
          │
          ▼
  Sélection de features (Cox PH, p-value < 0.01)
          │
          ▼
  Split 80/20 train/test
          │
          ▼
  Entraînement du modèle de survie
          │
          ▼
  Évaluation : C-index IPCW + Integrated Brier Score
          │
          ▼
  Sauvegarde modèle + logs JSON (dossier horodaté)
```

---

## Modèles supportés

| Modèle | Description |
|---|---|
| `CoxPHSurvivalAnalysis` | Régression de Cox — modèle semi-paramétrique (défaut) |
| `RandomSurvivalForest` | Forêt aléatoire adaptée aux données censurées |
| `GradientBoostingSurvivalAnalysis` | Gradient Boosting pour la survie |

---

## Métriques d'évaluation

- **Concordance Index IPCW** (tau = 7 ans) — mesure la capacité ordinale du modèle
- **Integrated Brier Score** (horizon 0–7 ans) — mesure l'erreur de calibration au fil du temps

---

## Stack technique

- **Survie** : `scikit-survival >= 0.22.0`, `lifelines >= 0.27.0`
- **Données** : `polars`, `pandas`, `pyarrow`
- **Features** : `category_encoders`, `mygene` (annotation génomique)

---

## Installation

```bash
git clone https://github.com/motawassimlahmadi/leukemia-survival-analysis.git
cd leukemia-survival-analysis
pip install -r requirements.txt
```

Ou via le script fourni :

```bash
bash install_dependencies.sh
```

---

## Utilisation

### Lancer un modèle spécifique

```bash
python main.py \
  --dataset_path="data/raw/X_train/molecular_train.csv" \
  --ml_method="CoxPHSurvivalAnalysis" \
  --ml_params="{}"
```

### Lancer tous les modèles en séquence

```bash
bash run_all_methods.sh
```

### Mode low_memory (réduit la taille des fichiers modèle de 5–11 Go)

```bash
python main.py \
  --dataset_path="data/raw/X_train/molecular_train.csv" \
  --ml_method="RandomSurvivalForest" \
  --ml_params='{"low_memory": true}'
```

> **Note** : le mode `low_memory` désactive le calcul de l'Integrated Brier Score.

---

## Arguments CLI

| Argument | Description | Défaut |
|---|---|---|
| `--dataset_path` | Chemin vers le fichier CSV | `""` |
| `--ml_method` | Modèle de survie à utiliser | `CoxPHSurvivalAnalysis` |
| `--ml_params` | Hyperparamètres du modèle (dict JSON) | `{}` |
| `--save_dir` | Dossier de sortie (modèle + logs) | `models/` |

---

## Sorties

Chaque exécution génère un dossier horodaté dans `save_dir/` contenant :
- Le modèle entraîné (`.pkl`)
- Un fichier de configuration JSON
- Un fichier de logs avec les métriques
