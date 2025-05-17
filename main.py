import argparse
import os
import json
from time import gmtime, strftime

from src.data.molecular_preprocess import process_molecular_data
from src.data.load_data import load_dataset
from src.data.clinical_preprocess import process_clinical_data
from src.data.y_train_preprocess import y_train_preprocess
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv

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

    # Molecular data set
    df= load_dataset(args.dataset_path)
    df = process_molecular_data(df)
    
    for col in df.columns:
        if df.select(pl.col(col).is_null().sum())[0, 0] > 0:
            print(f"Nom de colonne avec valeurs nulles : {col} , Nombre : {df.select(pl.col(col).is_null().sum())[0, 0]}")
    
    # Clinical data set
    cl_df = load_dataset("data/raw/X_train/clinical_train.csv")
    cl_df = process_clinical_data(cl_df)
    
    for col in cl_df.columns:
        if cl_df.select(pl.col(col).is_null().sum())[0, 0] > 0:
            print(f"Nom de colonne avec valeurs nulles : {col} , Nombre : {cl_df.select(pl.col(col).is_null().sum())[0, 0]}")
    
    
    # Joining the two datasets
    df_mol_cl = df.join(cl_df , on="ID" , how="left")
    
    # Y_train data set
    y_train = load_dataset("data/raw/target_train.csv")
    y_train = y_train_preprocess(y_train)
    
    
    final_df = df_mol_cl.join(y_train , on="ID" , how="left")
    
    final_df = final_df.drop("ID")
    final_df = final_df.drop([col for col, dtype in zip(final_df.columns, final_df.dtypes) if dtype == pl.Utf8])
    
    print(final_df)
    
    df_pd = final_df.to_pandas()
    
    
    for col in final_df.columns:
        if final_df.select(pl.col(col).is_null().sum())[0, 0] > 0:
            print(f"Nom de colonne avec valeurs nulles : {col} , Nombre : {final_df.select(pl.col(col).is_null().sum())[0, 0]}")


    # Variables d'entrée
    X = df_pd.drop(columns=["OS_STATUS", "OS_YEARS"])
    y_class = df_pd["OS_STATUS"]
    y_reg = df_pd["OS_YEARS"]

    # Split
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_class_train)
    y_class_pred = clf.predict(X_test)
    acc = accuracy_score(y_class_test, y_class_pred)
    print(f"Accuracy OS_STATUS: {acc:.4f}")
    
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train, y_reg_train)
    y_reg_pred = reg.predict(X_test)
    rmse = root_mean_squared_error(y_reg_test, y_reg_pred)
    print(f"RMSE OS_YEARS: {rmse:.4f}")
    
    
    

if __name__ == "__main__":
    main()