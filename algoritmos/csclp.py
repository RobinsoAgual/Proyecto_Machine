import numpy as np
import pulp
from sklearn.metrics.pairwise import cosine_distances

def aplicar_csclp(X, k=5, size_constraint=30):
    n = len(X)
    distancias = cosine_distances(X)
    problema = pulp.LpProblem("CSCLP", pulp.LpMinimize)

    # Variables x[i][c]: punto i pertenece al cluster c
    x = [[pulp.LpVariable(f"x_{i}_{c}", cat="Binary") for c in range(k)] for i in range(n)]

    # Variables y[i][j][c] solo para i < j
    y = {}
    for i in range(n):
        for j in range(i+1, n):
            for c in range(k):
                y[i, j, c] = pulp.LpVariable(f"y_{i}_{j}_{c}", cat="Binary")

    # Objetivo: minimizar distancia entre pares en el mismo cluster
    problema += pulp.lpSum(distancias[i][j] * y[i, j, c]
                           for i in range(n)
                           for j in range(i+1, n)
                           for c in range(k))

    # Restricción: cada punto en un solo cluster
    for i in range(n):
        problema += pulp.lpSum(x[i][c] for c in range(k)) == 1

    # Restricción: límite de tamaño por cluster
    for c in range(k):
        problema += pulp.lpSum(x[i][c] for i in range(n)) <= size_constraint

    # Restricciones de consistencia entre x e y
    for i in range(n):
        for j in range(i+1, n):
            for c in range(k):
                problema += y[i, j, c] <= x[i][c]
                problema += y[i, j, c] <= x[j][c]
                problema += y[i, j, c] >= x[i][c] + x[j][c] - 1

    # Resolver
    problema.solve(pulp.PULP_CBC_CMD(msg=0))

    # Extraer etiquetas
    etiquetas = [-1] * n
    for i in range(n):
        for c in range(k):
            if pulp.value(x[i][c]) == 1:
                etiquetas[i] = c
                break

    return etiquetas
