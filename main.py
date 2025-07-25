import streamlit as st
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics.pairwise import cosine_distances

# Importaciones específicas de tu proyecto
from utils.buscador_dataset_openml import cargar_dataset_openml, buscar_datasets_utiles
from utils.preprocesamiento_texto import limpiar_texto, tokenizar_texto
from embeddings.deepseek_embedding import calcular_deepseek
from embeddings.gemini_embedding import calcular_gemini
from embeddings.qwen_embedding import calcular_qwen
from embeddings.tfidf_embedding import generar_tfidf
from embeddings.word2vec_embedding import calcular_word2vec
from algoritmos import (
    kmeans_ba,
    csclp,
    aco,
    hcakc,
    kmedoids,
    pso,
    km_milp,
    sck1,
)
from evaluacion.metricas_clustering import evaluar_clustering
from evaluacion.visualizaciones import graficar_clusters_pca, graficar_silueta, graficar_matriz_confusion
from utils.matching import mapear_clusters_a_clases

# Configuración inicial de la página
st.set_page_config(page_title="Clustering Interactivo", layout="wide")
st.title("📊 Clustering Interactivo con Embeddings y Algoritmos")

# Cargar datasets útiles desde OpenML
@st.cache_data
def cargar_datasets():
    return buscar_datasets_utiles(min_instancias=100, max_instancias=1000, max_features=20, limite=10)


datasets_utiles = cargar_datasets()

# Selección de dataset
st.header("📦 Selección de Dataset")
dataset_seleccionado = st.selectbox(
    "Selecciona un dataset:",
    options=[ds['name'] for ds in datasets_utiles],
    index=0,
)
did_seleccionado = next(ds["did"] for ds in datasets_utiles if ds["name"] == dataset_seleccionado)

# Cargar el dataset seleccionado
textos_crudos, textos_limpios, etiquetas, columna_texto, columna_label = cargar_dataset_openml(did_seleccionado)
n_clusters_default = len(set(etiquetas))
st.success(f"✅ Dataset cargado: columna de texto = '{columna_texto}' | clase = '{columna_label}'")

# Selección de embedding
st.header("🔍 Selección de Embedding")
embedding_options = {
    "TFIDF": generar_tfidf,
    "WORD2VEC": calcular_word2vec,
    "GEMINI": calcular_gemini,
    "DEEPSEEK": calcular_deepseek,
    "QWEN": calcular_qwen,
}
embedding_seleccionado = st.selectbox(
    "Selecciona un método de embedding:",
    options=list(embedding_options.keys()),
    index=0,
)

# Generar embeddings
if st.button("Generar Embeddings"):
    with st.spinner("📏 Generando embeddings..."):
        funcion_embed = embedding_options[embedding_seleccionado]
        X_embed = None

        # Si es QWEN, pedimos la API key
        if embedding_seleccionado == "QWEN":
            api_key = st.text_input("ac51997257ec4458837bb25a720fea9f")
            if not api_key:
                st.warning("⚠️ Debes ingresar una API Key válida.")
            else:
                try:
                    X_embed = funcion_embed(textos_limpios, api_key)
                except Exception as e:
                    st.error(f"❌ Error generando embeddings con Qwen: {e}")
        else:
            try:
                X_embed = funcion_embed(textos_limpios)
            except Exception as e:
                st.error(f"❌ Error generando embeddings: {e}")

        if X_embed is not None:
            # Verificar duplicados
            unique_rows, indices = np.unique(X_embed, axis=0, return_index=True)
            num_duplicados = len(textos_limpios) - len(unique_rows)
            if num_duplicados > 0:
                st.warning(f"⚠️ Se encontraron {num_duplicados} puntos duplicados en los embeddings.")

            # Verificar varianza cero
            variances = np.var(X_embed, axis=0)
            if np.any(variances == 0):
                st.warning("⚠️ Algunas dimensiones tienen varianza cero. Agregando ruido leve...")
                X_embed += np.random.normal(scale=1e-8, size=X_embed.shape)

            # Preprocesamiento adicional
            scaler = StandardScaler()
            X_embed = scaler.fit_transform(X_embed)

            # Validar que haya suficientes instancias para PCA
            if X_embed.shape[0] < 2:
                st.warning("⚠️ No se pueden aplicar PCA con menos de 2 instancias.")
            else:
                # Aplicar PCA automáticamente según varianza acumulada
                pca_full = PCA()
                X_pca_full = pca_full.fit_transform(X_embed)
                varianza_acumulada = np.cumsum(pca_full.explained_variance_ratio_)

                # Elegir número mínimo de componentes para explicar al menos el 95% de la varianza
                num_componentes_optimos = np.argmax(varianza_acumulada >= 0.95) + 1

                # Aplicar PCA con ese número de componentes
                pca = PCA(n_components=num_componentes_optimos)
                X_embed = pca.fit_transform(X_embed)

                st.success(f"✅ PCA aplicado automáticamente con {num_componentes_optimos} componentes (≥95% varianza explicada)")


            st.session_state["X_embed"] = X_embed
            st.success("✅ Embeddings generados correctamente.")

# Selección de algoritmo de clustering
st.header("⚙️ Selección de Algoritmo de Clustering")
algoritmo_options = {
    "KMEANS": kmeans_ba.aplicar_kmeans_ba,
    "ACO": aco.aplicar_aco,
    "CSCLP": csclp.aplicar_csclp,
    "HC-AKC": hcakc.aplicar_hcakc,
    "KMEDOIDS": kmedoids.aplicar_kmedoids_sc,
    "KM-MILP": km_milp.aplicar_km_milp,
    "PSO": pso.aplicar_pso_clustering,
    "SCK1": sck1.aplicar_sck1,
}
algoritmo_seleccionado = st.selectbox(
    "Selecciona un algoritmo de clustering:",
    options=list(algoritmo_options.keys()),
    index=0,
)

# Configuración automática de parámetros
st.subheader("🧠 Parámetros Calculados Automáticamente")
params = {}

# Determinar número de clústeres según clases verdaderas
n_clusters = len(set(etiquetas))
params["n_clusters"] = n_clusters
st.info(f"🔢 Número de clústeres (k) determinado automáticamente: {n_clusters}")

# Total de instancias
total_instancias = len(textos_limpios)

# Restricciones de tamaño por clúster (si aplica)
if algoritmo_seleccionado in ["SCK1", "KMEDOIDS", "KM-MILP", "PSO"]:
    default_size = total_instancias // n_clusters
    size_constraints = [default_size] * n_clusters
    remainder = total_instancias - sum(size_constraints)
    for i in range(remainder):
        size_constraints[i] += 1  # Repartir el sobrante
    params["size_constraints"] = size_constraints
    st.success(f"📏 Restricciones de tamaño asignadas automáticamente: {size_constraints}")

# Restricción de tamaño máxima para CSCLP
if algoritmo_seleccionado == "CSCLP":
    size_constraint = int(np.ceil(total_instancias / n_clusters))
    params["size_constraint"] = size_constraint
    st.success(f"📦 Tamaño máximo por clúster (CSCLP): {size_constraint}")

# Parámetros automáticos para PSO
if algoritmo_seleccionado == "PSO":
    params["swarmsize"] = 10
    params["max_iter"] = 20
    st.success("🧬 PSO configurado automáticamente con swarmsize=10, max_iter=20")


# Ejecutar clustering
if st.button("Ejecutar Clustering"):
    if "X_embed" not in st.session_state:
        st.error("❌ Primero genera los embeddings.")
    else:
        with st.spinner("📊 Ejecutando algoritmo de clustering..."):
            X_embed = st.session_state["X_embed"]
            num_instancias = X_embed.shape[0]

            # Validar n_clusters
            if params.get("n_clusters", 0) > num_instancias:
                st.error(f"❌ El número de clústeres ({params.get('n_clusters', 0)}) no puede ser mayor que el número de instancias ({num_instancias}).")
            elif params.get("n_clusters", 0) <= 1:
                st.error("❌ El número de clústeres debe ser mayor a 1.")
            else:
                # Filtrar parámetros según los que acepta la función
                funcion_clustering = algoritmo_options[algoritmo_seleccionado]
                parametros_filtrados = {
                    k: v for k, v in params.items() if k in funcion_clustering.__code__.co_varnames
                }

                # Ejecutar el algoritmo de clustering
                try:
                    etiquetas_pred = funcion_clustering(X_embed, **parametros_filtrados)
                except Exception as e:
                    st.error(f"❌ Error ejecutando el algoritmo de clustering: {e}")
                    st.stop()

                # Evaluar resultados
                etiquetas = [str(e) for e in etiquetas]
                etiquetas_pred = [str(e) for e in etiquetas_pred]
                resultados = evaluar_clustering(X_embed, etiquetas_pred, y_true=etiquetas)

                # Mostrar resultados en una tabla
                st.subheader("📋 Resultados del Clustering")
                st.table(resultados)

                # Visualización PCA
                st.subheader("📈 Visualización PCA")
                fig_pca = graficar_clusters_pca(X_embed, etiquetas_pred)
                st.pyplot(fig_pca)

                # Visualización de Silueta (si se puede)
                try:
                    X_embed = np.squeeze(X_embed)
                    if X_embed.ndim != 2:
                        X_embed = X_embed.reshape(X_embed.shape[0], -1)

                    if len(etiquetas_pred) != X_embed.shape[0]:
                        raise ValueError("Longitud de etiquetas no coincide con las instancias de embeddings.")

                    from sklearn.metrics import silhouette_samples
                    from sklearn.metrics.pairwise import cosine_distances

                    D = cosine_distances(X_embed)
                    silhouette_values = silhouette_samples(D, etiquetas_pred, metric="precomputed")

                    fig_silhouette = graficar_silueta(silhouette_values, etiquetas_pred, params.get("n_clusters", 2), "Gráfico de Silueta")
                    st.pyplot(fig_silhouette)
                except Exception as e:
                    st.warning(f"⚠️ No se pudo generar el gráfico de silueta: {e}")

                # Visualización de Matriz de Confusión
                etiquetas_pred_mapeadas = mapear_clusters_a_clases(etiquetas, etiquetas_pred)
                st.subheader("🧩 Matriz de Confusión")
                try:
                    etiquetas_unicas = sorted(list(set(etiquetas)))
                    fig_cm = graficar_matriz_confusion(etiquetas, etiquetas_pred_mapeadas)
                    st.pyplot(fig_cm)
                except Exception as e:
                    st.warning(f"⚠️ No se pudo generar la matriz de confusión: {e}")
