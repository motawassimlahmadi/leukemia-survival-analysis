import polars as pl
import numpy as np
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from src.data.basic_preprocess import *
import re



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


def cyto_regex(cyto , reg_expr):
    if cyto is not None:
        if re.search(reg_expr,cyto,re.IGNORECASE):
            return 1
    return 0



# Used for inv , del , add 
def anomaly_number(cytogenetic , keyword , regex_expr):
    if cyto_regex(cytogenetic ,keyword):
        del_nbr = re.findall(regex_expr, cytogenetic)
        if del_nbr: 
            return int(del_nbr[0])
    return -1 


# DETECTION DE TRANSLOCATION

def transloc_nbr(cytogenetic):
    if cyto_regex(cytogenetic, 't'):
        trans_nbr = re.findall(r"t\((\d+);(\d+)\)", cytogenetic)
        if trans_nbr:
            flat = [int(x) for tup in trans_nbr for x in tup]
            return flat
    return [-1]

def process_clinical_data(cl_df):
    
    # Validate input
    if not isinstance(cl_df, pl.DataFrame):
        raise TypeError("Input must be a polars DataFrame")
    
    # Outliers
    cl_df = iqr_method(cl_df, ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"])
    
    
    # 2. Imputation of missing values for VAF and DEPTH
    cl_df = imputation_null_values(cl_df, ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"], RandomForestRegressor())
    
    # CENTER Encoding
    cl_df = binary_encoder(cl_df, ["CENTER"])
    
    # Male Karyotype
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_a_Man" , is_male_karyotype , pl.Int32)
    
    # Female karyotype
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_a_Female" , is_female_karyotype , pl.Int32)
    
    # Deletion anomaly
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_deletion_anomaly" , cyto_regex(reg_expr=r'del'), pl.Int32)
    
    # Deleted chromosome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "deleted_chromosome" , anomaly_number(keyword=r'del' , regex_expr=r"del\((\d+.*)\)"), pl.Int64)
    
    # Inversion anomaly
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_inversion_anomaly" , cyto_regex(reg_expr=r'inv' ) , pl.Int32)
    
    # Inverted chromosme
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "deleted_chromosome" , anomaly_number(keyword=r'inv' , regex_expr=r"inv\((\d+.*)\)"), pl.Int64)
    
    # Added chromosome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_added_anomaly" , cyto_regex(reg_expr=r'add' ) , pl.Int32)
    
    # Added chromosme
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "added_chromosome" , anomaly_number(keyword=r'add' , regex_expr=r"add\((\d+.*)\)"), pl.Int64)
    
    # Translocation anomaly
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_translocated_anomaly" , cyto_regex(reg_expr=r't' ) , pl.Int32)
    
    # Translocated anomaly
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "translocated_chromosome" , transloc_nbr, pl.List(pl.Int64))
    
    # Downs Syndrome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_down_syndrome" , cyto_regex(keyword=r'\+21' ) , pl.Int32)
    
    # Monosomy 7 
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_monosomy" , cyto_regex(keyword=r'\-7' ) , pl.Int32)
    
    
    # Partial deletion of 7th chromosome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_7_deleted" , cyto_regex(keyword=r'del\(7\)' ) , pl.Int32)
    
    # Trisomy 8
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_trisomy_8" , cyto_regex(keyword=r'\+8' ) , pl.Int32)
    
    # Isochromosome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "iso_chromosome" , cyto_regex(keyword=r"i\(\d+\)|iso\(\d.+\)") , pl.Int64)
    
    # Derived chromosome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_derived_chromosome" , cyto_regex(keyword=r"der\(\d+\)") , pl.Int64)
    
    # Lost sex chromosome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_lost_sex_chromosome" , cyto_regex(keyword=r"\-[xy]") , pl.Int64)
    
    # Added chromosomes
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_added_chromsome" , cyto_regex(keyword=r",\+\d+") , pl.Int64)
    
    # Deleted chromosomes
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_deleted_chromsome" , cyto_regex(keyword=r",\-\d+") , pl.Int64)
    
    # Inserted chromosomes
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_inserted_chromsome" , cyto_regex(keyword=r"ins") , pl.Int64)
    
    
    # Min-max Normalization
    
    col_to_normalize  = ["BM_BLAST", "WBC", "ANC", "HB", "PLT"]
    
    cl_df = min_max_normalization(cl_df , col_to_normalize)
    
    return cl_df
    
    
    
    
    
    
    
    
    


