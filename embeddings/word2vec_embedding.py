# embeddings/word2vec_embedding.py
import numpy as np
import spacy

# Cargar el modelo preentrenado de spaCy
print("📦 Cargando modelo de spaCy...")
nlp = spacy.load("en_core_web_md")

def calcular_word2vec(textos):
    """
    Calcula embeddings usando spaCy.
    
    Args:
        textos (list[str]): Lista de textos.
    
    Returns:
        np.ndarray: Matriz de embeddings, donde cada fila corresponde a un texto.
    """
    print("🔍 Generando embeddings con spaCy...")
    embeddings = []
    for texto in textos:
        # Procesar el texto con spaCy
        doc = nlp(texto)
        # Obtener el vector promedio de las palabras en el texto
        if doc.vector.any():
            embeddings.append(doc.vector)
        else:
            embeddings.append(np.zeros(nlp.meta["vectors"]["width"]))  # Vector de ceros si no hay palabras conocidas
    return np.array(embeddings)