import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def aplicar_kmeans_ba(X, n_clusters=5, seeds=10):
    """
    Aplica KMeans buscando la mejor semilla (Best of Seeds) usando Silhouette Score.
    """
    if n_clusters <= 0:
        raise ValueError("El número de clústeres debe ser mayor que 0.")
    
    mejor_score = -np.inf
    mejor_modelo = None

    for semilla in range(seeds):
        modelo = KMeans(n_clusters=n_clusters, random_state=semilla, n_init='auto')
        etiquetas = modelo.fit_predict(X)
        if len(set(etiquetas)) > 1:  # Evitar errores si hay un solo cluster
            score = silhouette_score(X, etiquetas)
            if score > mejor_score:
                mejor_score = score
                mejor_modelo = modelo

    if mejor_modelo is None:
        raise ValueError("No se pudo encontrar un modelo válido con más de un cluster. Revisa tus datos o parámetros.")
    
    return mejor_modelo.labels_
