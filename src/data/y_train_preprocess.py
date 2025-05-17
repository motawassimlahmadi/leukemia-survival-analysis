import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from src.data.basic_preprocess import *
import category_encoders as ce
from sklearn.ensemble import RandomForestRegressor
from sksurv.util import Surv



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

def y_train_surv(df):
    df = df.with_columns(
        pl.col("OS_STATUS").cast(pl.Boolean).alias("OS_STATUS")
    )
    
    df = df.rename(
        {"OS_YEARS" : "time" ,
        "OS_STATUS" : "event"}
    )
    
    y_train_pd = df.to_pandas()

    y_train_pd = Surv.from_dataframe('event' , 'time' , y_train_pd)
    
    return y_train_pd





    