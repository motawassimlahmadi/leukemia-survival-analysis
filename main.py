import argparse
import os
import json
from time import gmtime, strftime

from src.data.load_data import load_dataset, load_additional_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train_model import split_data, train_model, save_model_and_results
from src.models.clustering import  apply_opt_kmeans, apply_kmeans , silouhette_score , apply_label , centroids , cluster_mean , describe_clusters
from src.visualization.visualize import plot_pca

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
    
    print(df)
    
    

if __name__ == "__main__":
    main()