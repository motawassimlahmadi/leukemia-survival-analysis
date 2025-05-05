import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from src.data.basic_preprocess import *
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor



def y_train_preprocess(df):
    
    # Imputation Null Values
    df = imputation_null_values(df, [df.columns[1]], estimator=RandomForestRegressor())
    
    # IQR METHOD
    df = iqr_method(df , ["OS_YEARS"])
    
    # Null values
    df = df.with_columns(
        pl.col("OS_STATUS").fill_null(strategy="backward")
    )
    
    return df





    