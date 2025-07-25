import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics.pairwise import cosine_distances

def aplicar_hcakc(X, n_clusters=5, method='ward'):
    """
    Aplica HC-AKC: Hierarchical Clustering with Assignment and Cardinality Knowledge Constraints.
    """
    if n_clusters <= 0:
        raise ValueError("El número de clústeres debe ser mayor que 0.")
    
    distancia = cosine_distances(X)
    linkage_matrix = linkage(distancia, method=method)
    labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    return labels - 1  # Ajuste para comenzar desde 0