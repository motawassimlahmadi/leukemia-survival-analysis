import polars as pl
import numpy as np
from sklearn.cluster import KMeans
from src.features.build_features import build_features
from matplotlib import pyplot as plt
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
import pandas as pd


def apply_kmeans(X):
    inertia = []
    k_range = range(2, 20)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    kl = KneeLocator(k_range, inertia, curve='convex', direction='decreasing')
    optimal_k = kl.knee 
    
    return optimal_k


def apply_opt_kmeans(X , k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred= kmeans.fit_predict(X)
    
    
    return (y_pred,kmeans)

def apply_label(df,kmeans):
    df = df.with_columns(
        pl.lit(kmeans.labels_).alias("Cluster")
    )
    
    return df

def silouhette_score(X, kmeans):
    return silhouette_score(X, kmeans.labels_)

def centroids(df):
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    
    features = df.drop(columns='Cluster', errors='ignore').columns
    k = df['Cluster'].nunique()
    kmeans = KMeans(n_clusters=k, random_state=42).fit(df[features])
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=features)
    importance = centroids.var().sort_values(ascending=False)
    
    
    
    return importance

def size_cluster(df):
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    
    return df['Cluster'].value_counts().sort_index()

def cluster_mean(df):
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    cluster_means = df.groupby('Cluster').mean(numeric_only=True).round(2)
    
    return cluster_means

def cluster_analysis(df, col, n=3):
    return df.sort_values(by=col, ascending=False).head(n)[[col]]



def describe_clusters(cluster_means, top_n=3):
    descriptions = []

    # Normaliser les valeurs par colonne pour détecter ce qui ressort
    norm = (cluster_means - cluster_means.mean()) / cluster_means.std()

    for cluster_id, row in norm.iterrows():
        original = cluster_means.loc[cluster_id]
        desc = f"\n🔹 **Cluster {cluster_id}**\n"

        # Identifier les features les plus hautes/faibles
        top_features = row.sort_values(ascending=False).head(top_n).index
        low_features = row.sort_values().head(top_n).index

        # Ajouter descriptions textuelles
        desc += "📈 Points forts : " + ", ".join([
            f"{feat} ({original[feat]:.1f})" for feat in top_features
        ]) + "\n"

        desc += "📉 Points faibles : " + ", ".join([
            f"{feat} ({original[feat]:.1f})" for feat in low_features
        ]) + "\n"

        # Suggestion de profil (simple heuristique)
        if "Mood After Workout" in top_features or "Workout Duration (mins)" in top_features:
            profile = "Sportif motivé et endurant 💪"
        elif "Mood Before Workout" in low_features or "Workout Duration (mins)" in low_features:
            profile = "Fatigué ou peu impliqué 💤"
        elif "CB / Duration" in top_features:
            profile = "Très performant sur le plan physique 🔥"
        else:
            profile = "Profil intermédiaire ou équilibré 🧘"

        desc += f"🧠 **Profil suggéré** : {profile}\n"
        descriptions.append(desc)

    return "\n".join(descriptions)

    