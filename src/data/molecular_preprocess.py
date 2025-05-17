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
import category_encoders as ce
import mygene
from sklearn.preprocessing import MultiLabelBinarizer
import re
from src.data.basic_preprocess import *




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



def gene_new_name(df , cl):
    # MLL => KMT2A
    # WHSC1 => NSD2
    # H3F3A => H3-3A 
    # FAM175A => ABRAXAS1
    # PAPD5 => TENT4B

    # NATIONAL LIBRARY OF MEDICINE

    mapping = {'MLL' : 'KMT2A' , 'WHSC1' : 'NSD2' , 'H3F3A' : 'H3-3A' , 'FAM175A' : 'ABRAXAS1' , 'PAPD5' : 'TENT4B'}

    df = df.with_columns(
        pl.col(cl).map_elements(lambda g: mapping.get(g, g)).alias(cl)
    )
    
    return df


def cytogenetics(df , cl):
    genes = list(df[cl].unique())

    mg = mygene.MyGeneInfo()

    results = mg.querymany(genes, scopes="symbol", fields='go', species="human")

    return results


def genes_to_go(results_cyto):
    gene_to_go = {}
    for res in results_cyto:
        gene = res.get('query')
        go_bp = res.get('go', {}).get('BP', [])
        if isinstance(go_bp, dict):  # Cas d'un seul terme
            go_bp = [go_bp]
        go_terms = [go['id'] for go in go_bp if 'id' in go]
        if gene and go_terms:
            gene_to_go[gene] = go_terms
            
    return gene_to_go


def multi_label_gene_go(gene_to_go):
    # 4. Créer une matrice binaire gène × GO term
    mlb = MultiLabelBinarizer()
    go_matrix = mlb.fit_transform(gene_to_go.values())
    go_df = pd.DataFrame(go_matrix, index=gene_to_go.keys(), columns=mlb.classes_)

    # 5. Filtrer les GO terms peu fréquents (seuil = 5 gènes)
    min_gene_count = 5
    filtered_go_df = go_df.loc[:, (go_df.sum(axis=0) >= min_gene_count)]

    
    return filtered_go_df


def merge_df(df1 , filtered_go_df , cl , how):
    df_pd = df1.to_pandas()

    df_merged = df_pd.merge(filtered_go_df, left_on=cl, right_index=True, how=how)
    
    df_merged = pl.from_pandas(df_merged)
    
    return df_merged



def protein_name(protein_changes, ch):
    lst_prot = []
    pattern = rf"p\.{ch}\d+.*"

    for prot in protein_changes:
        if isinstance(prot, str): 
            prot_name = re.findall(pattern, prot)
            if prot_name:
                lst_prot+=prot_name
                
    return lst_prot


def get_protein_dico(uppercase_alphabet , lst , aa_fullname):
    dico_protein_changes = {}
        
    for ch in uppercase_alphabet:
        if(len(lst) >=1):
            ch_name = aa_fullname.get(ch)
            dico_protein_changes[ch_name] = lst
            
    return dico_protein_changes


def protein_type(df , protein_change , dico_protein_changes):
    acide_aminee = {}

    for elem in protein_change:
        for aa_name , mutations in dico_protein_changes.items():
            if elem in mutations:
                acide_aminee[elem] = aa_name
                
                          
    # KMT2A => MLL PTD => Lysine 
    # FLT3-ID => FLT3 => tyrosine

    acide_aminee["MLL_PTD"] = "Lysine"
    acide_aminee["FLT3_ITD"] = "Tyrosine"
    acide_aminee["p.?"] = "Unknown"
    
    df = df.with_columns(
        pl.col("PROTEIN_CHANGE").map_elements(lambda x: acide_aminee.get(x, None)).alias("AA_TYPE")
    )
                
    return acide_aminee , df


def remove_row_element(df , col , lst):

    for elem in lst:
        df = df.remove(pl.col(col) == elem)
        
    return df


def drop_null_subset(df, subset):
    df = df.drop_nulls(subset=subset)
    
    return df


def is_protein_unknown(df):
    df = df.with_columns(
        pl.when(pl.col("PROTEIN_CHANGE") == "p.?")
        .then(1)
        .otherwise(0)
        .alias("IS_PT_UKNOWN")
    )
    
    return df

def is_frameshift(df):
    df = df.with_columns(
        pl.when(pl.col("PROTEIN_CHANGE").str.contains('fs'))
        .then(1)
        .otherwise(0)
        .alias("is_frameshift")
    )
    
    return df

def is_non_sens(df):
    #NON_SENS_MUTATION
    df = df.with_columns(
        pl.when(pl.col("PROTEIN_CHANGE").str.contains("\*"))
        .then(1)
        .otherwise(0)
        .alias("is_non_sens_mutation")
    )
    
    return df


def ismissense(elem):
    if elem is not None and isinstance(elem, str):
        # Vérifie que le dernier caractère est une lettre majuscule (acide aminé)
        if elem.startswith("p.") and elem[-1].isalpha() and "*" not in elem and "fs" not in elem:
            return 1
    return 0


def is_SNV(df , col , letter):
    df = df.with_columns(
        pl.when(pl.col(col) == letter)
        .then(1)
        .otherwise(0)
        .alias(f"{col}_is_SNV_{letter}")
    )
    
    return df




def complex_nucleotide(df , col ):
    df = df.with_columns(
        pl.when(pl.col(col).str.len_chars() > 1)
        .then(1)
        .otherwise(0)
        .alias(f"complex_nucleotide_{col}")
    )
    
    return df


def is_transition(str1, str2):
    if isinstance(str1, str) and isinstance(str2, str):
        if (str1 == "A" and str2 == "G") or (str1 == "G" and str2 == "A"):
            return 1
        elif (str1 == "C" and str2 == "T") or (str1 == "T" and str2 == "C"):
            return 1
    return 0

def is_transversion(str1, str2):
    if isinstance(str1, str) and isinstance(str2, str):
        trans_vers = {
            ("A", "C"), ("C", "A"), ("C", "G"), ("G", "C"),
            ("G", "T"), ("T", "G"), ("A", "T"), ("T", "A")
        }
        return int((str1, str2) in trans_vers)
    return 0

def is_indel(str1, str2):
    if isinstance(str1, str) and isinstance(str2, str):
        return int(len(str1) != len(str2))
    return 0



def add_mutation_density_features(mol_df):
    """Add mutation density features to the molecular dataframe"""

    # Count mutations per gene
    gene_counts = mol_df.group_by("GENE").agg(
        pl.len().alias("mutations_per_gene")
    )

    # Count mutations per gene and effect type
    gene_effect_counts = mol_df.group_by(["GENE", "EFFECT"]).agg(
        pl.len().alias("mutations_per_gene_effect")
    )

    # Count mutations per chromosome region (binned)
    region_counts = mol_df.group_by(["CHR", "START"]).agg(
        pl.len().alias("mutations_per_region")
    )

    # Join these counts back to the original dataframe
    mol_df = mol_df.join(gene_counts, on="GENE", how="left")

    # For gene-effect counts, we need to join on both columns
    mol_df = mol_df.join(gene_effect_counts, on=["GENE", "EFFECT"], how="left")
    
    # Join region counts
    mol_df = mol_df.join(region_counts, on=["CHR", "START"], how="left")

    return mol_df


def process_molecular_data(mol_df):
    
    # Validate input
    if not isinstance(mol_df, pl.DataFrame):
        raise TypeError("Input must be a polars DataFrame")
    
    # Outliers
    mol_df = iqr_method(mol_df , ["VAF" , "DEPTH" , "START" , "END"])

    # 1. Convert CHR to integer
    # X => 23
    mol_df = chr_to_int(mol_df, "CHR")
    
    # 2. Imputation of missing values for VAF and DEPTH
    mol_df = imputation_null_values(mol_df, ["CHR" , "START" , "END" , "VAF", "DEPTH"], RandomForestRegressor())


    # Add mutation density features to the molecular dataframe
    mol_df = add_mutation_density_features(mol_df)
    
    
    # 3. Apply gene name mapping
    mol_df = gene_new_name(mol_df, "GENE")
    
    
    # 4. Apply cytogenetics to GENE column
    results_cyto = cytogenetics(mol_df, "GENE")


    # 5. Process gene ontology data
    gene_to_go_dict = genes_to_go(results_cyto)
    filtered_go_df = multi_label_gene_go(gene_to_go_dict)

    # 6. Merge the dataframes
    mol_df = merge_df(mol_df, filtered_go_df, "GENE", "left")

    # 8. Process protein change information
    # Define amino acid mapping for protein analysis
    uppercase_alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    aa_fullname = {
        'A': 'Alanine', 'C': 'Cysteine', 'D': 'Aspartic_Acid', 'E': 'Glutamic_Acid',
        'F': 'Phenylalanine', 'G': 'Glycine', 'H': 'Histidine', 'I': 'Isoleucine',
        'K': 'Lysine', 'L': 'Leucine', 'M': 'Methionine', 'N': 'Asparagine',
        'P': 'Proline', 'Q': 'Glutamine', 'R': 'Arginine', 'S': 'Serine',
        'T': 'Threonine', 'V': 'Valine', 'W': 'Tryptophan', 'Y': 'Tyrosine'
    }
    
    
    # NULL PROTEIN CHANGE
    mol_df = mol_df.with_columns(
        pl.when(pl.col("PROTEIN_CHANGE").is_null())
        .then(pl.lit("p.?"))
        .otherwise(pl.col("PROTEIN_CHANGE"))
        .alias("PROTEIN_CHANGE")
    )

    # Process protein changes for each amino acid
    protein_changes = mol_df["PROTEIN_CHANGE"].to_list()
    dico_protein_changes = {}

    for ch in uppercase_alphabet:
        lst_prot = protein_name(protein_changes, ch)
        if lst_prot:
            ch_name = aa_fullname.get(ch)
            dico_protein_changes[ch_name] = lst_prot

    # Apply protein type classification
    acide_aminee, mol_df = protein_type(mol_df, protein_changes, dico_protein_changes)

    # 9. Remove specific protein changes
    protein_remove = ["p.*636C", "p.*342*", "p.*342S", "p.*636W"]
    mol_df = remove_row_element(mol_df, "PROTEIN_CHANGE", protein_remove)

    # 10. Mark unknown proteins
    mol_df = is_protein_unknown(mol_df)

    # 11. Apply binary encoding to AA_TYPE
    mol_df = binary_encoder(mol_df, ["AA_TYPE" , "EFFECT"])

    # 12. Identify frameshift mutations
    mol_df = is_frameshift(mol_df)

    # 13. Identify nonsense mutations
    mol_df = is_non_sens(mol_df)

    # 14. Identify missense mutations - optimized approach
    # Using the provided ismissense function directly with is_miss_sense
    mol_df = map_lambda(mol_df,"PROTEIN_CHANGE","IS_MISSENSE" , ismissense , pl.Int32)
    
    # Min_Max Normalization
    
    col_to_normalize = ["START", "END" , "DEPTH" , "mutations_per_gene" , "mutations_per_gene_effect" , "mutations_per_region"]
    
    
    #REF
    snv_lst = ['A','C','G','T']
    for snv in snv_lst:
        mol_df =  is_SNV(mol_df , "REF" , snv)
    
    #ALT
    for snv in snv_lst:
        mol_df =  is_SNV(mol_df , "ALT" , snv)
        
    # Complex Nucleotide
    
    mol_df = complex_nucleotide(mol_df , "REF")
    mol_df = complex_nucleotide(mol_df , "ALT")
    
    
    mol_df = map_row(mol_df, "REF", "ALT", "is_transition", is_transition, pl.Int8)
    mol_df = map_row(mol_df, "REF", "ALT", "is_transversion", is_transversion, pl.Int8)
    mol_df = map_row(mol_df, "REF", "ALT", "is_indel", is_indel, pl.Int8)

    
        
        
    mol_df = min_max_normalization(mol_df , col_to_normalize)
    
    
    

    return mol_df
    




    



