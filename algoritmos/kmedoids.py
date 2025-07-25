import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def pam_initial_medoids(D, k):
    """
    Selecciona medoides iniciales usando el algoritmo PAM.
    """
    n = D.shape[0]
    current_medoids = [np.random.randint(0, n)]
    while len(current_medoids) < k:
        dist_to_nearest = np.min(D[:, current_medoids], axis=1)
        probs = dist_to_nearest / dist_to_nearest.sum()
        next_medoid = np.random.choice(n, p=probs)
        current_medoids.append(next_medoid)
    return current_medoids

def sc_medoids(D, k, E, C):
    """
    Asigna puntos a clusters respetando restricciones de tamaño.
    """
    cl = np.argmax(-D[:, C], axis=1)
    sorted_points = np.argsort(np.min(D[:, C], axis=1))
    for i in range(k):
        cl[sorted_points[:E[i]]] = i
        sorted_points = sorted_points[E[i]:]
    for point in sorted_points:
        cl[point] = np.argmin(D[point, C])
    return cl

def aplicar_kmedoids_sc(X, size_constraints):
    """
    Aplica KMedoids con restricciones de tamaño.
    """
    k = len(size_constraints)
    if sum(size_constraints) != X.shape[0]:
        raise ValueError("La suma de las restricciones de tamaño debe coincidir con el número de instancias.")
    
    D = cosine_distances(X)
    medoids = pam_initial_medoids(D, k)
    etiquetas = sc_medoids(D, k, size_constraints, medoids)
    return etiquetas