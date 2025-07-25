import numpy as np
from scipy.optimize import linprog
from sklearn.metrics.pairwise import cosine_distances

def solve_milp_assignment(data, centroids, size_constraints):
    n, k = data.shape[0], centroids.shape[0]
    cost_vector = cosine_distances(data, centroids).flatten()

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

    result = linprog(cost_vector, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not result.success:
        raise ValueError("MILP no encontró solución.")

    return np.argmax(result.x.reshape(n, k), axis=1)

def aplicar_km_milp(X, size_constraints, max_iter=100, tol=1e-6):
    n, d = X.shape
    k = len(size_constraints)
    rng = np.random.default_rng(42)
    indices = rng.choice(n, size=k, replace=False)
    centroids = X[indices]

    for _ in range(max_iter):
        prev_centroids = centroids.copy()
        labels = solve_milp_assignment(X, centroids, size_constraints)
        for j in range(k):
            puntos = X[labels == j]
            if len(puntos) > 0:
                centroids[j] = puntos.mean(axis=0)
        if np.linalg.norm(centroids - prev_centroids) < tol:
            break

    return labels