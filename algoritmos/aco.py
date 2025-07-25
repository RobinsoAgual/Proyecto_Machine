import numpy as np
import random
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

def generar_cardinalidad_balanceada(n_instancias, n_clusters):
    """
    Genera una lista de cardinalidades balanceadas para los clusters.
    """
    base = n_instancias // n_clusters
    resto = n_instancias % n_clusters
    cardinalidades = [base] * n_clusters
    for i in range(resto):
        cardinalidades[i] += 1
    return cardinalidades

def aplicar_aco(X, n_clusters=3, n_ants=50, max_iterations=20, q=0.5, penalty_weight=100):
    """
    Implementación del algoritmo ACO para clustering con restricciones de tamaño.
    Devuelve las etiquetas del mejor agrupamiento encontrado.
    """
    n_instancias = X.shape[0]
    if n_clusters > n_instancias:
        raise ValueError("El número de clústeres no puede ser mayor que el número de instancias.")
    
    target_cardinality = generar_cardinalidad_balanceada(n_instancias, n_clusters)
    D = cosine_distances(X)

    mejor_puntaje = -np.inf
    mejor_etiquetado = None

    for iteracion in range(max_iterations):
        hormigas = [generar_solucion_inicial(X, n_clusters, target_cardinality) for _ in range(n_ants)]
        hormigas = [perturbar_solucion(asignacion, n_clusters, target_cardinality) for asignacion in hormigas]

        for asignacion in hormigas:
            puntaje = evaluar_solucion(X, asignacion, target_cardinality, penalty_weight)
            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_etiquetado = asignacion

        for asignacion in hormigas:
            actualizar_feromonas(D, asignacion, q, target_cardinality)

    return mejor_etiquetado


# Funciones auxiliares
def generar_solucion_inicial(X, n_clusters, target_cardinality):
    """
    Genera una solución inicial para el algoritmo ACO.
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X)
    labels = kmeans.labels_

    for j in range(n_clusters):
        while np.sum(labels == j) > target_cardinality[j]:
            idx = np.where(labels == j)[0]
            labels[np.random.choice(idx)] = np.random.randint(0, n_clusters)
    return labels


def perturbar_solucion(asignacion, n_clusters, target_cardinality):
    """
    Perturba una solución existente para explorar nuevas soluciones.
    """
    nueva_asignacion = np.copy(asignacion)
    n = len(asignacion)
    for j in range(n):
        if np.random.rand() < 0.1:
            nueva_asignacion[j] = np.random.randint(0, n_clusters)

    for j in range(n_clusters):
        while np.sum(nueva_asignacion == j) > target_cardinality[j]:
            idx = np.where(nueva_asignacion == j)[0]
            nueva_asignacion[np.random.choice(idx)] = np.random.randint(0, n_clusters)
    return nueva_asignacion


def evaluar_solucion(X, asignacion, target_cardinality, penalty_weight=100):
    """
    Evalúa una solución basada en el coeficiente de silueta y penalizaciones.
    """
    try:
        ss_values = silhouette_samples(X, asignacion, metric="cosine")
        ss_mean = np.mean(ss_values)
    except:
        ss_mean = -1  # Si hay un solo cluster o error

    penalty = 0
    for j in range(len(target_cardinality)):
        diff = abs(np.sum(asignacion == j) - target_cardinality[j])
        penalty += diff * penalty_weight
    return ss_mean - penalty


def actualizar_feromonas(D, asignacion, q, target_cardinality):
    """
    Actualiza las feromonas basadas en la calidad de la solución.
    """
    freqs = [np.sum(asignacion == j) for j in range(len(target_cardinality))]
    diff = sum(abs(np.array(freqs) - np.array(target_cardinality)))
    pheromone_update = q / (1 + diff)

    for j in range(len(asignacion)):
        D[j, :] *= (1 - pheromone_update)