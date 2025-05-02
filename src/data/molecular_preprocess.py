import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored , concordance_index_ipcw
from sklearn.impute import SimpleImputer
from sksurv.util import Surv

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



def to_int(str):
    if(str == 'X'):
        return 23 # A justifier ?
    else:
        return int(str)
    
def chr_to_int(df , col="CHR"):
    df = df.with_columns(
        pl.col(col).map_elements(to_int , return_dtype=pl.Int64)
        .alias(col)
    )
    
    return df

def dna_to_array(dna):
    lst = []
    
    nitrogen_bases = ['a','g','t','c']
    dna = dna.lower()
    
    
    for ch in dna:
        if(ch in nitrogen_bases):
            lst.append(ch)
        else:
            lst.append('n')
        
        
    return lst

def ordinal_encoder_dna(dna):
    mapping = {'a' : 0.25 , 'c' : 0.5 ,'g':0.75 , 't' : 1.00}
    
    nitrogen_bases = ['a','g','t','c']
    
    return [mapping[x] if x in nitrogen_bases else 0.00 for x in dna ]



def extract_kmers(sequence, k):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def join_str(str):
    return ' '.join(str)



