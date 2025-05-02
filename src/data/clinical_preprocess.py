import polars as pl
import numpy as np
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import category_encoders as ce
import re





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



def forward_imputation(df):
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



def binary_encoder(df, cl):
    df_pd = df.to_pandas()

    encoder = ce.BinaryEncoder(cols=[cl])
    df_encoded = encoder.fit_transform(df_pd)

    df = pl.from_pandas(df_encoded)
    
    return df



def is_male_karyotype(karyotype, male_code='46,xy'):
    return 1 if male_code in karyotype.lower() else 0

def is_female_karyotype(karyotype, female_code='46,xx'):
    return 1 if female_code in karyotype.lower() else 0

def is_a_Man(df , col):
    df = df.with_columns(
        pl.col(col).map_elements(lambda x: is_male_karyotype(x))
        .alias("is_a_Man")
    )
    
    return df

def is_a_Female(df , col):
    df = df.with_columns(
        pl.col(col).map_elements(lambda x: is_female_karyotype(x))
        .alias("is_a_Female")
    )
    
    return df



def contains_cyto(cytogenetic, keyword):
    return 1 if keyword in cytogenetic.lower() else 0


# Used for inv , del , add 
def anomaly_number(cytogenetic , keyword , regex_expr):
    if contains_cyto(cytogenetic ,keyword):
        del_nbr = re.findall(regex_expr, cytogenetic)
        if del_nbr: 
            return int(del_nbr[0])
    return -1 


# DETECTION DE TRANSLOCATION

def transloc_nbr(cytogenetic):
    if contains_cyto(cytogenetic, 't'):
        trans_nbr = re.findall(r"t\((\d+);(\d+)\)", cytogenetic)
        if trans_nbr:
            flat = [int(x) for tup in trans_nbr for x in tup]
            return flat
    return [-1]


def min_max_normalization(df , col):
    min = df[col].min()
    max = df[col].max()
    
    
    df = df.with_columns(
        ((pl.col(col) - min)/(max - min)).alias(f"{col}_min_max")
    )
    
    return df

def Z_scaling(df , col):
    mean = df[col].mean()
    std = df[col].std()
    
    
    df = df.with_columns(
        ((pl.col(col) - mean)/(std)).alias(f"{col}_Z_score")
    )
    
    return df


    