
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.impute import IterativeImputer
import category_encoders as ce




def map_lambda(df , col , new_name , function , return_type):
    df = df.with_columns(
        pl.col(col).map_elements(lambda x: function(x) , return_dtype=return_type)
        .alias(new_name)
    )
    
    return df


def map_row(df, col1, col2, new_name, function, return_type):
    return df.with_columns(
        pl.struct([col1, col2]).map_elements(
            lambda row: function(row[col1], row[col2]), return_dtype=return_type
        ).alias(new_name)
    )




def binary_encoder(df, cl_lst):
    df_pd = df.to_pandas()
    
    encoder = ce.BinaryEncoder(cols=cl_lst)
    df_encoded = encoder.fit_transform(df_pd)
    
    return pl.from_pandas(df_encoded)



def one_hot_encoder(df, cl_lst):
    # Convertir en pandas
    df_pd = df.to_pandas()

    # Appliquer OneHotEncoder de category_encoders
    encoder = ce.OneHotEncoder(cols=cl_lst, use_cat_names=True)
    df_encoded = encoder.fit_transform(df_pd)

    # Reconvertir en polars
    return pl.from_pandas(df_encoded)





def strategy_imputation(df , strategy):
    df = df.fill_null(strategy=strategy)

    return df





def imputation_null_values(df, column_list, estimator=None):
    """
    Impute missing values using iterative imputation.

    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame containing columns with missing values
    column_list : list
        List of column names to impute
    estimator : object, default=None
        Estimator for IterativeImputer (defaults to RandomForestRegressor if None)

    Returns:
    --------
    pl.DataFrame
        DataFrame with imputed values

    Raises:
    -------
    ValueError
        If input is not a Polars DataFrame or columns don't exist
    """
    if not isinstance(df, pl.DataFrame):
        raise ValueError("Input must be a Polars DataFrame")

    missing_columns = [col for col in column_list if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for imputation: {', '.join(missing_columns)}")

    # Check if there are any null values to impute
    null_counts = {col: df.select(pl.col(col).is_null().sum())[0, 0] for col in column_list}
    if sum(null_counts.values()) == 0:
        return df  # No nulls to impute

    try:
        # Select subset of dataframe
        sub_df = df.select(column_list)

        # Convert to pandas
        sub_pd = sub_df.to_pandas()
        

        # Apply imputation
        imputer = IterativeImputer(estimator=estimator, random_state=42)
        imputed_data = imputer.fit_transform(sub_pd)

        # Convert back to polars
        imputed_pl = pl.DataFrame(imputed_data, schema=sub_df.columns)

        # Replace nulls in original dataframe
        df = df.with_columns([imputed_pl[col].alias(col) for col in column_list])

        return df
    except Exception as e:
        raise ValueError(f"Error during imputation: {str(e)}")



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

def min_max_normalization(df, column_list):
    """
    Apply min-max normalization to specified columns.

    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame containing columns to normalize
    column_list : list
        List of column names to normalize

    Returns:
    --------
    pl.DataFrame
        DataFrame with normalized columns

    Raises:
    -------
    ValueError
        If input is not a Polars DataFrame or columns don't exist
    """
    if not isinstance(df, pl.DataFrame):
        raise ValueError("Input must be a Polars DataFrame")

    missing_columns = [col for col in column_list if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns for normalization: {', '.join(missing_columns)}")

    try:
        for col in column_list:
            min_val = df[col].min()
            max_val = df[col].max()

            # Avoid division by zero
            if min_val == max_val:
                df = df.with_columns(
                    pl.lit(0.5).alias(col)
                )
            else:
                df = df.with_columns(
                    ((pl.col(col) - min_val)/(max_val - min_val)).alias(col)
                )

        return df
    except Exception as e:
        raise ValueError(f"Error during min-max normalization: {str(e)}")

def Z_scaling(df , col):
    mean = df[col].mean()
    std = df[col].std()
    
    
    df = df.with_columns(
        ((pl.col(col) - mean)/(std)).alias(f"{col}_Z_score")
    )
    
    return df
    