import argparse
import os
import json
from time import gmtime, strftime

from src.data.molecular_preprocess import process_molecular_data
from src.data.load_data import load_dataset
from src.data.clinical_preprocess import process_clinical_data

def main():
    
    # Arguments
    parser = argparse.ArgumentParser(
        prog="24/25 ML Project Example",
        description="Example program for the ML Project course (2024/2025 M1 IDD)"
    )

    parser.add_argument("--dataset_path", type=str, default="", help="path to the dataset file")
    parser.add_argument("--ml_method", type=str, default="Linear", help="name of the ML method to use ('XGBoost', 'LightGBM', 'Logistic Regression')")
    parser.add_argument("--l2_penalty", type=float, default=1., help="strength of the L2 penalty used when fitting the model")
    parser.add_argument("--cv_nsplits", type=int, default=5, help="cross-validation: number of splits")
    parser.add_argument("--save_dir", type=str, default="models", help="where to save the model, the logs and the configuration")

    args = parser.parse_args()

    # Create the directory containing the model, the logs, etc.
    dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    out_dir = os.path.join(args.save_dir, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    path_config = os.path.join(out_dir, "config.json")

    # Store the configuration
    with open(path_config, "w") as f:
        json.dump(vars(args), f)

    # Loading
    df= load_dataset(args.dataset_path)
    
    df = process_molecular_data(df)
    
    cl_df = load_dataset("data/raw/X_train/clinical_train.csv")
    
    cl_df = process_clinical_data(cl_df)
    
    print(df)
    print(cl_df)
    
    
    

if __name__ == "__main__":
    main()