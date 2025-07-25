import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from pyswarm import pso

def cost_function(par, dist_matrix, cluster_sizes, k):
    """
    Función de costo para PSO.
    """
    n = len(par)
    cluster_assignment = np.zeros(n, dtype=int)
    start_idx = 0
    sorted_indices = np.argsort(par)
    for i in range(k):
        end_idx = start_idx + cluster_sizes[i]
        cluster_assignment[sorted_indices[start_idx:end_idx]] = i
        start_idx = end_idx
    total_distance = 0
    for i in range(k):
        indices = np.where(cluster_assignment == i)[0]
        if len(indices) > 1:
            cluster_dist = dist_matrix[np.ix_(indices, indices)]
            total_distance += np.sum(np.tril(cluster_dist, -1))
    return total_distance

def aplicar_pso_clustering(X, n_clusters, size_constraints, swarmsize, max_iter):
    """
    Aplica PSO Clustering con restricciones de tamaño.
    """
    k = len(size_constraints)
    if sum(size_constraints) != X.shape[0]:
        raise ValueError("La suma de las restricciones de tamaño debe coincidir con el número de instancias.")
    
    D = cosine_distances(X)
    lb = np.zeros(X.shape[0])
    ub = np.full(X.shape[0], k)
    args = (D, size_constraints, k)

    best, _ = pso(cost_function, lb, ub, args=args, swarmsize=swarmsize, maxiter=max_iter)

    sorted_indices = np.argsort(best)
    labels = np.zeros(X.shape[0], dtype=int)
    start = 0
    for i in range(k):
        end = start + size_constraints[i]
        labels[sorted_indices[start:end]] = i
        start = end
    return labels