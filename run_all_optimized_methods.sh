#!/bin/bash

DATASET_PATH="data/raw/X_train/molecular_train.csv"


ML_METHODS=("CoxPHSurvivalAnalysis" "RandomSurvivalForest" "GradientBoostingSurvivalAnalysis")


ML_PARAMS=(
  '{"alpha": 1, "n_iter": 100, "ties": "breslow"}'
  '{"max_features": "sqrt", "min_samples_leaf": 3, "min_samples_split": 6, "n_estimators": 200}'
  '{"learning_rate": 1, "max_depth": 5, "n_estimators": 200}'
)


for i in "${!ML_METHODS[@]}"; do
  echo "Running with method: ${ML_METHODS[$i]}"
  python main.py --dataset_path="$DATASET_PATH" --ml_method="${ML_METHODS[$i]}" --ml_params="${ML_PARAMS[$i]}"
  echo "---------------------------------------------"
done
