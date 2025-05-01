import polars as pl
import numpy as np
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor



def imputation_null_values(df , cl , estimator) :
    quant_vars = cl
    sub_df = df.select(quant_vars)

    # Convert to pandas
    sub_pd = sub_df.to_pandas()

    # Apply Model-Based Imputation (Random Forest)
    imputer = IterativeImputer(estimator=estimator, random_state=42)
    imputed_data = imputer.fit_transform(sub_pd)

    # 4. To Polars
    imputed_pl = pl.DataFrame(imputed_data, schema=sub_df.columns)

    # 5. Replace nulls
    df = df.with_columns([imputed_pl[col].alias(col) for col in quant_vars])
    
    return df



def forward_imputation(df , cl ):
    df = df.fill_null(strategy="forward")

    return df


def iqr_method(df , cl_lst):

    for (index , cl )  in enumerate(cl_lst):
        q1 = df.select(pl.col(cl).quantile(0.25)).item()
        q3 = df.select(pl.col(cl).quantile(0.75)).item()
        
        iqr = q3 - q1
        
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        
        # Winsorisation : capped values
        df = df.with_columns(
            pl.when(pl.col(cl) < lower_bound).then(lower_bound)
            .when(pl.col(cl) > upper_bound).then(upper_bound)
            .otherwise(pl.col(cl))
            .alias(cl)
        )
        
        
        
        return df
    

def z_score(df, cl_list):
    
    for cl in cl_list:
        mean = df.select(pl.col(cl).mean()).item()
        std = df.select(pl.col(cl).std()).item()

        # Définir les bornes avec Z = ±3
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std

        # Appliquer la winsorisation
        df = df.with_columns(
            pl.when(pl.col(cl) < lower_bound).then(lower_bound)
            .when(pl.col(cl) > upper_bound).then(upper_bound)
            .otherwise(pl.col(cl))
            .alias(cl)
        )

    return df
