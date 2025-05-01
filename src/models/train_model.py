from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import json
import os
import pickle
from time import gmtime, strftime

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Train test split
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def get_param_grid(ml_method):
    """
    Returns the param grid for a specific ml_method
    """
    param_grid = {
        "XGBoost": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 9]
        },
        "LightGBM": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 9]
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"]
        }
    }

    return param_grid.get(ml_method, {})

def get_model(ml_method):
    """
    Returns the ml_method specified
    """
    if ml_method == "XGBoost":
        return xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    elif ml_method == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    elif ml_method == "Logistic Regression":
        return LogisticRegression(max_iter=10000, random_state=42)
    else:
        raise ValueError("Méthode ML non reconnue. Choisissez parmi 'XGBoost', 'LightGBM' ou 'Logistic Regression'.")

def train_model(X_train, y_train, X_test, y_test, ml_method, cv_nsplits=5):
    """
    Trains the model with the specified ml_method
    """
    model = get_model(ml_method)
    param_grid = get_param_grid(ml_method)

    # Polars to pandas for compatibility reasons
    if ml_method == "LightGBM":
        X_train = X_train.to_pandas() if hasattr(X_train, 'to_pandas') else X_train
        X_test = X_test.to_pandas() if hasattr(X_test, 'to_pandas') else X_test
        y_train = y_train.to_pandas() if hasattr(y_train, 'to_pandas') else y_train
        y_test = y_test.to_pandas() if hasattr(y_test, 'to_pandas') else y_test

    # Searching for the best parameters
    grid = GridSearchCV(model, param_grid, cv=cv_nsplits, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    # Best parameters and model
    best_params = grid.best_params_
    best_model = grid.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)

    # Scores
    score_mse = mean_squared_error(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    acc_train = accuracy_score(y_train, y_pred_train)

    results = {
        "best_params": best_params,
        "score_mse": score_mse,
        "accuracy_score": acc_score,
        "accuracy_train": acc_train
    }

    return best_model, results

def save_model_and_results(model, results, save_dir=""):
    """
    Saves the model and the results
    """
    # Création du répertoire de sortie
    dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    out_dir = os.path.join(save_dir, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # Files directory
    path_model = os.path.join(out_dir, "model.pkl")
    path_logs = os.path.join(out_dir, "logs.json")

    # Saving the model
    with open(path_model, "wb") as f:
        pickle.dump(model, f)

    # Saving the results
    with open(path_logs, "w") as f:
        json.dump(results, f)

    return out_dir