# embeddings/deepseek_embedding.py
import numpy as np
from sentence_transformers import SentenceTransformer

print("📦 Cargando modelo local de embeddings (MiniLM)...")

try:
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo preentrenado
except Exception as e:
    raise ImportError("Error al cargar modelo local (MiniLM): pip install sentence-transformers")

def calcular_deepseek(textos):
    """
    Calcula embeddings usando un modelo local (MiniLM).
    
    Args:
        textos (list[str]): Lista de textos.
    
    Returns:
        np.ndarray: Matriz de embeddings.
    """
    print("🔍 Calculando embeddings con modelo local...")
    try:
        embeddings = model.encode(textos, show_progress_bar=True)
        return np.array(embeddings)
    except Exception as e:
        print(f"⚠️ Error calculando embeddings: {e}")
        return np.zeros((len(textos), 384))  # Embedding de fallback