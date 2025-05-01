import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.decomposition import PCA


def plot_correlation_matrix(df):
    """
    Correlation matrix
    """
    matrix = df.corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(matrix, cmap='Blues')

    plt.colorbar()
    variables = matrix.columns

    plt.xticks(range(len(matrix)), variables, rotation=45, ha='right')
    plt.yticks(range(len(matrix)), variables)

    plt.title('Correlation matrix')
    plt.tight_layout()

    return plt.gcf()



def cluster_mean(df):
    cluster_means = df.groupby('Cluster').mean(numeric_only=True).round(2)
    
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(cluster_means.T, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Moyenne des features par cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()
    


def plot_pca(X, kmeans):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.title("Projection PCA des clusters")
    plt.xlabel("Composante principale 1")
    plt.ylabel("Composante principale 2")
    plt.colorbar()
    plt.show()

    



def plot_feature_importance(importance):
    plt.figure(figsize=(10, 6))
    importance.plot(kind='barh')
    plt.title("Importance des features pour le clustering")
    plt.xlabel("Variance inter-cluster")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
        

def plot_feature_importance(model, feature_names):
    """
    Features importances
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Importance des caractéristiques')
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()

        return plt.gcf()
    else:
        print("Le modèle ne possède pas d'attribut feature_importances_")
        return None

def plot_class_distribution(y):
    """
    Shows the class distribution
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Distribution des classes')
    plt.xlabel('Classe')
    plt.ylabel('Nombre d\'échantillons')

    return plt.gcf()

def plot_learning_curve(model, X, y, cv=5):
    """
    Shows the learning curve
    """
    from sklearn.model_selection import learning_curve

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='r', label='Train score')
    plt.plot(train_sizes, test_mean, 'o-', color='g', label='Validation score')

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')

    plt.xlabel('Size of the set of training')
    plt.ylabel('Score')
    plt.title('Learning curve')
    plt.legend(loc='best')
    plt.grid(True)

    return plt.gcf()