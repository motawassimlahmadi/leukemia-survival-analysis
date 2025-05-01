import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA


def head(df):
    
    return df.head()


def describe(df):
    return df.describe()

def histplot(df , cl):
    cl_quantitative_var = cl

    rows = 2
    cols = 3

    plt.figure(figsize=(15, 8)) 

    for index, var in enumerate(cl_quantitative_var):
        plt.subplot(rows, cols, index + 1)
        sns.histplot(data=df, x=var, bins=15, kde=True)
        plt.title(var)

    plt.tight_layout()
    plt.show()
    

def number_center(df):
    sns.countplot(data=df,x="CENTER")
    plt.xticks(rotation=45)
    plt.title("Number of center")
    plt.show()
    

        