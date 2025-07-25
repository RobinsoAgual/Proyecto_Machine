import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_samples
)
from sklearn.metrics.pairwise import cosine_distances


def evaluar_clustering(X, pred, y_true):
    """
    Evalúa las métricas de clustering.
    
    Args:
        X: Matriz de características.
        pred: Etiquetas predichas.
        y_true: Etiquetas reales.
    
    Returns:
        dict: Diccionario con las métricas calculadas.
    """
    # Verificar si hay al menos 2 clusters
    n_clusters = len(set(pred))
    if n_clusters < 2:
        return {
            "ARI": None,
            "AMI": None,
            "NMI": None,
            "Silhouette_promedio": None,
            "num_clusters": n_clusters
        }

    # Calcular métricas
    D = cosine_distances(X)
    sil = silhouette_samples(D, pred)

    resultados = {
        "ARI": adjusted_rand_score(y_true, pred),
        "AMI": adjusted_mutual_info_score(y_true, pred),
        "NMI": normalized_mutual_info_score(y_true, pred),
        "Silhouette_promedio": np.mean(sil),
        "num_clusters": n_clusters
    }

    return resultados