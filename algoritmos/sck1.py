import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from scipy.optimize import linprog

def solve_ilp_assignment(cost_matrix, size_constraints):
    """
    Resuelve el problema ILP para asignar puntos a clusters.
    """
    k, n = cost_matrix.shape
    c = cost_matrix.flatten()

    Aeq1 = np.zeros((n, n * k))
    for i in range(n):
        Aeq1[i, i * k:(i + 1) * k] = 1
    beq1 = np.ones(n)

    Aeq2 = np.zeros((k, n * k))
    for j in range(k):
        Aeq2[j, j::k] = 1
    beq2 = size_constraints

    Aeq = np.vstack([Aeq1, Aeq2])
    beq = np.concatenate([beq1, beq2])
    bounds = [(0, 1)] * (n * k)

    result = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not result.success:
        raise ValueError("ILP no encontró solución.")

    x_opt = result.x.reshape(n, k)
    return np.argmax(x_opt, axis=1)

def aplicar_sck1(X, size_constraints, max_iter=100, tol=1e-6):
    """
    Aplica SCK1 con restricciones de tamaño.
    """
    n, d = X.shape
    k = len(size_constraints)
    if sum(size_constraints) != n:
        raise ValueError("La suma de las restricciones de tamaño debe coincidir con el número de instancias.")
    
    rng = np.random.default_rng(42)
    indices = rng.choice(n, size=k, replace=False)
    centroids = X[indices]

    for _ in range(max_iter):
        prev_centroids = centroids.copy()
        dist_mat = cosine_distances(X, centroids)
        labels = solve_ilp_assignment(dist_mat.T, size_constraints)
        for j in range(k):
            puntos = X[labels == j]
            if len(puntos) > 0:
                centroids[j] = puntos.mean(axis=0)
        if np.linalg.norm(centroids - prev_centroids) < tol:
            break

    return labels