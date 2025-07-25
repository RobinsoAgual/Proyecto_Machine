# embeddings/tfidf_manual_embedding.py
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def generar_tfidf(textos):
    """
    Calcula embeddings usando TF-IDF manualmente.
    
    Args:
        textos (list[str]): Lista de textos.
    
    Returns:
        np.ndarray: Matriz de embeddings.
    """
    print("📊 Calculando embeddings con TF-IDF manual...")
    try:
        vectorizer = CountVectorizer()
        bow = vectorizer.fit_transform(textos).toarray()
        
        # Calcular TF
        tf = np.zeros_like(bow, dtype=float)
        tf_mask = bow > 0
        tf[tf_mask] = 1 + np.log10(bow[tf_mask])

        
        # Calcular DF e IDF
        df = np.count_nonzero(bow, axis=0)
        idf = np.log10(len(bow) / (df + 1))
        
        # Calcular TF-IDF
        tfidf = tf * idf
        
        # Normalizar
        return tfidf / np.linalg.norm(tfidf, axis=1, keepdims=True)
    except Exception as e:
        print(f"⚠️ Error calculando TF-IDF manual: {e}")
        return np.zeros((len(textos), len(vectorizer.get_feature_names_out())))  # Embedding de fallback