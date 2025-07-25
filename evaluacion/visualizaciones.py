import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.decomposition import PCA

def graficar_clusters_pca(X, labels, titulo="Clusters en espacio PCA"):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Convertir etiquetas a enteros únicos para usar como colores
    etiquetas_unicas = sorted(list(set(labels)))
    etiquetas_map = {et: i for i, et in enumerate(etiquetas_unicas)}
    labels_int = np.array([etiquetas_map[et] for et in labels])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels_int,
        cmap="tab10",
        s=50
    )

    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.set_title(titulo)
    ax.grid(True)
    plt.tight_layout()
    return fig


def graficar_silueta(sil, pred, K=None, titulo="Gráfico de Silueta"):
    """
    Genera un gráfico de silueta robusto y flexible.

    Args:
        sil (array): Coeficientes de silueta.
        pred (array): Etiquetas de cluster predichas.
        K (int, opcional): Número de clusters. Si None, se infiere.
        titulo (str): Título del gráfico.

    Returns:
        fig: Objeto de figura matplotlib.
    """
    sil = np.array(sil)
    pred = np.array(pred)
    etiquetas_unicas = np.unique(pred)

    if K is None:
        K = len(etiquetas_unicas)

    MAX_PUNTOS_POR_CLUSTER = 1000
    fig, ax = plt.subplots(figsize=(10, 6))
    y_lower = 10

    for idx, etiqueta in enumerate(etiquetas_unicas):
        valores = sil[pred == etiqueta]
        if len(valores) == 0:
            continue

        if len(valores) > MAX_PUNTOS_POR_CLUSTER:
            valores = np.random.choice(valores, MAX_PUNTOS_POR_CLUSTER, replace=False)

        valores.sort()
        tam = len(valores)
        y_upper = y_lower + tam
        color = cm.nipy_spectral(float(idx) / K)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, valores, facecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * tam, str(etiqueta))
        y_lower = y_upper + 10

    ax.axvline(x=np.mean(sil), color="red", linestyle="--", label="Silhouette promedio")
    ax.set_title(titulo)
    ax.set_xlabel("Coeficiente de silueta")
    ax.set_ylabel("Cluster")
    ax.set_yticks([])
    ax.legend(loc="upper right")

    plt.tight_layout()
    return fig

def graficar_matriz_confusion(y_true, y_pred, etiquetas=None, titulo="Matriz de Confusión"):
    """
    Muestra la matriz de confusión con etiquetas ajustadas y diagonal resaltada.

    Args:
        y_true: Etiquetas reales
        y_pred: Etiquetas predichas
        etiquetas: Lista opcional de etiquetas personalizadas
        titulo: Título del gráfico
    """
    cm = confusion_matrix(y_true, y_pred, labels=etiquetas)
    etiquetas = etiquetas if etiquetas else np.unique(np.concatenate((y_true, y_pred)))

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.set(font_scale=0.8)

    # Mapa de calor sin normalizar
    matriz = ax.imshow(cm, interpolation='nearest', cmap="Greys")
    plt.title(titulo)
    tick_marks = np.arange(len(etiquetas))
    plt.xticks(tick_marks, etiquetas, rotation=90)
    plt.yticks(tick_marks, etiquetas)

    # Mostrar los valores y resaltar la diagonal
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            valor = cm[i, j]
            color = 'blue' if i == j else 'black'
            ax.text(j, i, str(valor), ha="center", va="center", color=color, fontsize=8, fontweight='bold' if i == j else 'normal')

    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.colorbar(matriz)
    plt.tight_layout()

    return fig