#!/bin/bash

DATASET_PATH="data/raw/X_train/molecular_train.csv"


ML_METHODS=("CoxPHSurvivalAnalysis" "RandomSurvivalForest" "GradientBoostingSurvivalAnalysis")


ML_PARAMS=(
  "{}"  # Params for CoxPHSurvivalAnalysis
  "{}"  # Example param for RandomSurvivalForest to reduce file size
  "{}"  # Params for GradientBoostingSurvivalAnalysis
)


for i in "${!ML_METHODS[@]}"; do
  echo "Running with method: ${ML_METHODS[$i]}"
  python main.py --dataset_path="$DATASET_PATH" --ml_method="${ML_METHODS[$i]}" --ml_params="${ML_PARAMS[$i]}"
  echo "---------------------------------------------"
done
