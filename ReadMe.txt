1) Install Dependencies

Run this command to install all required packages:

pip install -r requirements.txt


2) How to Launch the Code from an IDE

Run the main script with the following command format:

python main.py --dataset_path="data/raw/X_train/molecular_train.csv" --ml_method="{Valid ML Method}" --ml_params="{dict ML params}"

- --dataset_path: Path to the dataset CSV file.
- --ml_method: (Optional) Specify a valid machine learning method.
- --ml_params: (Optional) Provide a dictionary of parameters for the ML method.


Notes:

- If no ML method and no ML parameters are provided, the script will default to using the CoxPHSurvivalAnalysis model.
- The RandomSurvivalModel.pkl file generated can be very large (around 5GB to 11GB). To drastically reduce the file size, use the parameter:

  --ml_params='{"low_memory": true}'

- Important: If you enable low_memory=True, the integrated_brier_score evaluation will not work.
