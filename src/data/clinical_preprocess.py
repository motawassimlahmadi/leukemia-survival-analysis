import polars as pl
import numpy as np
import random
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from src.data.basic_preprocess import *
import re



def is_male_karyotype(karyotype, male_code='46,xy'):
    """
    Checks if the given karyotype string indicates a male.

    Parameters:
        karyotype (str): The karyotype string.
        male_code (str): The male code to search for (default: '46,xy').

    Returns:
        int: 1 if male code is found, otherwise 0.
    """
    return 1 if male_code in karyotype.lower() else 0


def is_female_karyotype(karyotype, female_code='46,xx'):
    """
    Checks if the given karyotype string indicates a female.

    Parameters:
        karyotype (str): The karyotype string.
        female_code (str): The female code to search for (default: '46,xx').

    Returns:
        int: 1 if female code is found, otherwise 0.
    """
    return 1 if female_code in karyotype.lower() else 0


def is_a_Man(df, col):
    """
    Adds a binary 'is_a_Man' column to the DataFrame based on the karyotype.

    Parameters:
        df (polars.DataFrame): The input DataFrame.
        col (str): Column name containing karyotype strings.

    Returns:
        polars.DataFrame: DataFrame with 'is_a_Man' column added.
    """
    df = df.with_columns(
        pl.col(col).map_elements(lambda x: is_male_karyotype(x))
        .alias("is_a_Man")
    )
    return df


def is_a_Female(df, col):
    """
    Adds a binary 'is_a_Female' column to the DataFrame based on the karyotype.

    Parameters:
        df (polars.DataFrame): The input DataFrame.
        col (str): Column name containing karyotype strings.

    Returns:
        polars.DataFrame: DataFrame with 'is_a_Female' column added.
    """
    df = df.with_columns(
        pl.col(col).map_elements(lambda x: is_female_karyotype(x))
        .alias("is_a_Female")
    )
    return df


def cyto_regex(cyto, reg_expr):
    """
    Checks if a given cytogenetic string matches a regex pattern.

    Parameters:
        cyto (str): The cytogenetic string.
        reg_expr (str): The regex expression to search for.

    Returns:
        int: 1 if pattern is found, otherwise 0.
    """
    if cyto is not None:
        if re.search(reg_expr, cyto, re.IGNORECASE):
            return 1
    return 0


def anomaly_number(cytogenetic, keyword, regex_expr):
    """
    Extracts chromosome number related to an anomaly (e.g., del, inv, add).

    Parameters:
        cytogenetic (str): Cytogenetic description.
        keyword (str): Anomaly keyword to look for.
        regex_expr (str): Regular expression to extract number.

    Returns:
        int: Extracted number if found, otherwise -1.
    """
    if cyto_regex(cytogenetic, keyword):
        del_nbr = re.findall(regex_expr, cytogenetic)
        if del_nbr:
            return int(del_nbr[0])
    return -1


def transloc_nbr(cytogenetic):
    """
    Detects and extracts chromosome numbers involved in translocations.

    Parameters:
        cytogenetic (str): Cytogenetic description.

    Returns:
        list: A list of chromosome numbers involved, or [-1] if none.
    """
    if cyto_regex(cytogenetic, 't'):
        trans_nbr = re.findall(r"t\((\d+);(\d+)\)", cytogenetic)
        if trans_nbr:
            flat = [int(x) for tup in trans_nbr for x in tup]
            return flat
    return [-1]


def is_complex(cyto):
    """
    Checks if a karyotype is complex based on number of abnormalities.

    Parameters:
        cyto (str): Cytogenetic string.

    Returns:
        int: 1 if complex (3+ abnormality keywords), otherwise 0.
    """
    return 1 if len(re.findall(r"\+|\-|del|t|inv|add|i\(", cyto)) >= 3 else 0


def is_missing_cytogenetics(df):
    """
    Replaces missing (null) values in the DataFrame with -1.

    Parameters:
        df (polars.DataFrame): The input DataFrame.

    Returns:
        polars.DataFrame: DataFrame with nulls replaced by -1.
    """
    for col in df.columns:
        null_count = df.select(pl.col(col).is_null().sum())[0, 0]
        if null_count > 0:
            df = df.with_columns(
                pl.when(pl.col(col).is_null()).then(-1).otherwise(pl.col(col)).alias(col)
            )
    return df


def process_clinical_data(cl_df):
    
    # Validate input
    if not isinstance(cl_df, pl.DataFrame):
        raise TypeError("Input must be a polars DataFrame")
    
    # Outliers
    cl_df = iqr_method(cl_df, ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"])
    
    
    # 2. Imputation of missing values for VAF and DEPTH
    cl_df = imputation_null_values(cl_df, ["BM_BLAST", "WBC", "ANC", "MONOCYTES", "HB", "PLT"], estimator=DecisionTreeRegressor())
    
    
    # # CENTER Encoding
    # cl_df = binary_encoder(cl_df, ["CENTER"])
    
    
    
    # Female karyotype
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_a_Female" , is_female_karyotype , pl.Int32)
    
    # Deletion anomaly
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_deletion_anomaly" , lambda x: cyto_regex(x, reg_expr=r'del'), pl.Int32)
    
    # Deleted chromosome
    
    # cl_df = map_lambda(cl_df , "CYTOGENETICS" , "deleted_chromosome" , lambda x: anomaly_number(x,keyword=r'del' , regex_expr=r"del\((\d+)\)"), pl.Int64)
    
    # Inversion anomaly
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_inversion_anomaly" , lambda x: cyto_regex(x , reg_expr=r'inv' ) , pl.Int32)
    
    # Inverted chromosme
    # cl_df = map_lambda(cl_df , "CYTOGENETICS" , "inverted_chromosome" , lambda x: anomaly_number(x,keyword=r'inv' , regex_expr=r"inv\((\d+)\)"), pl.Int64)
    
    # Added chromosome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_added_anomaly" , lambda x: cyto_regex(x , reg_expr=r'add' ) , pl.Int32)
    
    # Added chromosme
    # cl_df = map_lambda(cl_df , "CYTOGENETICS" , "added_chromosome" , lambda x: anomaly_number(x,keyword=r'add' , regex_expr=r"add\((\d+)\)"), pl.Int64)
    
    # Translocation anomaly
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_translocated_anomaly" , lambda x: cyto_regex(x ,reg_expr=r't' ) , pl.Int32)
    
    # # Translocated anomaly
    # cl_df = map_lambda(cl_df , "CYTOGENETICS" , "translocated_chromosome" , transloc_nbr, pl.List(pl.Int64))
    
    # Downs Syndrome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_down_syndrome" , lambda x: cyto_regex(x , reg_expr=r'\+21' ) , pl.Int32)
    
    # Monosomy 7 
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_monosomy" , lambda x: cyto_regex(x ,reg_expr=r'\-7' ) , pl.Int32)
    
    
    # Partial deletion of 7th chromosome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_7_deleted" , lambda x: cyto_regex(x ,reg_expr=r'del\(7\)' ) , pl.Int32)
    
    # Trisomy 8
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_trisomy_8" , lambda x: cyto_regex(x , reg_expr=r'\+8' ) , pl.Int32)
    
    # Isochromosome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "iso_chromosome" , lambda x: cyto_regex(x ,reg_expr=r"i\(\d+\)|iso\(\d.+\)") , pl.Int64)
    
    # Derived chromosome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_derived_chromosome" , lambda x: cyto_regex(x,reg_expr=r"der\(\d+\)") , pl.Int64)
    
    # Lost sex chromosome
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_lost_sex_chromosome" , lambda x: cyto_regex(x,reg_expr=r"\-[xy]") , pl.Int64)
    
    # Added chromosomes
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_added_chromsome" , lambda x: cyto_regex(x,reg_expr=r",\+\d+") , pl.Int64)
    
    # Deleted chromosomes
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_deleted_chromsome" , lambda x: cyto_regex(x,reg_expr=r",\-\d+") , pl.Int64)
    
    # Inserted chromosomes
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_inserted_chromsome" , lambda x: cyto_regex(x,reg_expr=r"ins") , pl.Int64)
    
    # Complex Karyotype
    cl_df = map_lambda(cl_df , "CYTOGENETICS" , "is_complex_karyo" , is_complex , pl.Int64)
    
    
    cl_df = is_missing_cytogenetics(cl_df)
    
    # Min-max Normalization
    
    col_to_normalize  = ["BM_BLAST", "WBC", "ANC", "HB", "PLT"]
    
    cl_df = min_max_normalization(cl_df , col_to_normalize)
    
    
    
    
    
    return cl_df
    
    
    
    
    
    
    
    
    


