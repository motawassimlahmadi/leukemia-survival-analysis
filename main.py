import argparse
import os
import json
from time import gmtime, strftime

from src.data.molecular_preprocess import process_molecular_data
from src.data.load_data import load_dataset
from src.data.clinical_preprocess import process_clinical_data
from src.data.y_train_preprocess import y_train_preprocess , y_train_surv
import polars as pl
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.ensemble import GradientBoostingSurvivalAnalysis ,  RandomSurvivalForest
import matplotlib.pyplot as plt
import numpy as np
from lifelines import CoxPHFitter
import seaborn as sns
from sklearn.model_selection import StratifiedKFold , GridSearchCV , KFold
from sksurv.metrics import integrated_brier_score
from lightgbm import LGBMRegressor
from sklearn.feature_selection import RFE
import ast
import joblib




def main():
    
    # Arguments
    parser = argparse.ArgumentParser(
        prog="24/25 ML Project Example",
        description= " ML Project course (2024/2025 M1 IDD)"
    )

    parser.add_argument("--dataset_path", type=str, default="", help="path to the dataset file")
    parser.add_argument("--ml_method", type=str, default="CoxPHSurvivalAnalysis", help="name of the ML method to use ('XGBoost', 'LightGBM', 'Logistic Regression')")
    parser.add_argument("--ml_params" , type=ast.literal_eval , default={} , help="The params to use for the ML method . Use a dictionnary for this . Example : {alpha : 0.01}")
    parser.add_argument("--save_dir", type=str, default="models", help="where to save the model, the logs and the configuration")

    args = parser.parse_args()

    # Create the directory containing the model, the logs, etc.
    dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    out_dir = os.path.join(args.save_dir, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    path_config = os.path.join(out_dir, "config.json")
    path_log = os.path.join(out_dir, "log.json")
    path_model = os.path.join(out_dir, "model.plk")
    

    # Store the configuration
    with open(path_config, "w") as f:
        json.dump(vars(args), f)
        
        
    # CHECKING ARGUMENTS
    arg_model  = args.ml_method
    arg_params = args.ml_params
    
    models = {"CoxPHSurvivalAnalysis" : CoxPHSurvivalAnalysis() , 
              "RandomSurvivalForest" : RandomSurvivalForest() , 
              "GradientBoostingSurvivalAnalysis" : GradientBoostingSurvivalAnalysis()
        }
    
    def validate_and_configure_model(model_name, param_dict, model_dict):
        if model_name not in model_dict:
            raise ValueError(f"Invalid model name '{model_name}'. Choose one of: {list(model_dict.keys())}")
    
        model = model_dict[model_name]
        valid_params = model.get_params()
        
        

        for key in param_dict:
            if key not in valid_params:
                raise ValueError(f"Unknown parameter '{key}' for model '{model_name}'")
            expected_type = type(valid_params[key])
            if valid_params[key] is not None and not isinstance(param_dict[key], expected_type):
                raise ValueError(f"Parameter '{key}' expects {expected_type}, got {type(param_dict[key])}")

        model.set_params(**param_dict)
        return model
    
    
    model = validate_and_configure_model(arg_model,arg_params,models)
    
    
    

    
    # PRE-PROCESSING

    # Molecular data set
    df= load_dataset(args.dataset_path)
    df = process_molecular_data(df)
    
    
    
    # Clinical data set
    cl_df = load_dataset("data/raw/X_train/clinical_train.csv")
    cl_df = process_clinical_data(cl_df)
    
    
    
    # Joining the two datasets
    df_mol_cl = df.join(cl_df , on="ID" , how="left")
    
    # Y_train data set
    y_train = load_dataset("data/raw/target_train.csv")
    y_train = y_train_preprocess(y_train)
    
    
    final_df = df_mol_cl.join(y_train , on="ID" , how="left")
    
    # Drops categorical columns
    final_df = final_df.drop("ID")
    final_df = final_df.drop([col for col, dtype in zip(final_df.columns, final_df.dtypes) if dtype == pl.Utf8])
    
    
    
    df_pd = final_df.to_pandas()
    
    def feature_selection(df):
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(df , duration_col='OS_YEARS', event_col='OS_STATUS')
        
        summary_df = cph.summary
        
        insignificant_vars = summary_df[summary_df['p'] > 0.01].index.tolist()
        
        df_significant = df.drop(columns=insignificant_vars)
        
        return df_significant
        
        
    
    df_pd = feature_selection(df_pd)
    
    
        
    
    # Choosing y target
    y_feature = final_df.select("OS_STATUS","OS_YEARS")
    
    
    X = df_pd.drop(["OS_STATUS" , "OS_YEARS"] , axis=1)
    y = y_train_surv(y_feature) # Formatting it to survival dataframe
    
    
    
    # Saving pre-processed dataframe
    pre_processed_molecular = df.write_csv("data/processed/X_train/molecular_train_preprocess.csv")
    pre_processed_clinical = cl_df.write_csv("data/processed/X_train/clinical_train_preprocess.csv")
    pre_processed_target = y_feature.write_csv("data/processed/target_preprocess.csv")
    final_processed_dataframe = df_pd.to_csv("data/processed/final_dataframe.csv")
    
    
    X_train , X_test , y_train_split , y_test_split = train_test_split(X , y, random_state=42, test_size=0.2)
    
    
    
    # Model Training
    model.fit(X_train, y_train_split)
    
    
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    
    
    # Concordance index IPCW score
    train_ci_ipcw = concordance_index_ipcw(y_train_split, y_train_split, pred_train, tau=7)[0]
    test_ci_ipcw = concordance_index_ipcw(y_train_split, y_test_split, pred_test, tau=7)[0]
    
    # Integrated Brier Score
    survs_train = model.predict_survival_function(X_train)
    survs_test = model.predict_survival_function(X_test)
    
    times = np.arange(0,7) # 0 to 7 years
    preds_train = np.asarray([[fn(t) for t in times] for fn in survs_train])
    preds_test = np.asarray([[fn(t) for t in times] for fn in survs_test])
    
    ib_score_train = integrated_brier_score(y_train_split,y_train_split,preds_train,times)
    ib_score_test = integrated_brier_score(y_train_split,y_test_split,preds_test,times)
    
    
    
    
    print(f"{arg_model} Survival Model Concordance Index IPCW on train: {train_ci_ipcw:.2f}")
    print(f"{arg_model} Survival Model Concordance Index IPCW on test: {test_ci_ipcw:.2f}")
    
    print(f"{arg_model} Integrated Brier Score on train: {ib_score_train:.4f}")
    print(f"{arg_model} Integrated Brier Score on test: {ib_score_test:.4f}")
    
    
    results = {
        "train_ci_ipcw": round(train_ci_ipcw, 2),
        "test_ci_ipcw": round(test_ci_ipcw, 2),
        "train_ib_score" : round(ib_score_train , 4),
        "test_ib_score" : round(ib_score_test , 4)
    }

    with open(path_log, "w") as f:
        json.dump(results, f, indent=2)  # indent for readability



    # Saving final model
    joblib.dump(model, f'{out_dir}/{arg_model}.pkl')

    
    
    
    
    
            
    
    
    

        
        
if __name__ == "__main__":
    main()