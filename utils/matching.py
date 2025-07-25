from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
import numpy as np

def mapear_clusters_a_clases(y_true, y_pred):
    """
    Mapea los clústeres predichos a las clases reales usando asignación óptima (Hungarian).
    """
    clases_reales = np.unique(y_true)
    clusters_predichos = np.unique(y_pred)

    # Confusion matrix alineada: filas = reales, columnas = predichos
    cm = confusion_matrix(y_true, y_pred, labels=clases_reales)
    
    # Hungarian algorithm
    cost_matrix = cm.max() - cm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Crear mapeo: cluster predicho → clase real
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        if j < len(clusters_predichos) and i < len(clases_reales):
            mapping[clusters_predichos[j]] = clases_reales[i]

    # Aplicar mapeo
    y_pred_mapeado = [mapping.get(c, c) for c in y_pred]

    return y_pred_mapeado
