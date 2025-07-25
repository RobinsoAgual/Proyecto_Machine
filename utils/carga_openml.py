import pandas as pd
import numpy as np
from utils.preprocesamiento_texto import limpiar_texto

def cargar_dataset_openml(did: int):
    """
    Carga un dataset desde OpenML y lo preprocesa.
    """
    # Intentar importar openml con manejo de errores
    try:
        import openml
    except ImportError as e:
        raise ImportError("OpenML no está instalado. Ejecuta 'pip install openml'") from e
    
    # Descargar el dataset
    try:
        dataset = openml.datasets.get_dataset(did)
        df, _, _, _ = dataset.get_data()
    except Exception as e:
        raise ConnectionError(f"Error al obtener dataset ID {did} de OpenML: {str(e)}")
    
    # Validar columna de clase
    class_col = dataset.default_target_attribute
    if class_col not in df.columns:
        raise ValueError(f"Columna de clase '{class_col}' no encontrada en el dataset")
    
    # Detectar columna de texto
    text_col = None
    text_candidates = {}

    # Identificar columnas candidatas
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col == class_col:
            continue
        try:
            sample = df[col].dropna().astype(str).sample(n=min(100, len(df)), random_state=42)
            word_counts = sample.apply(lambda x: len(x.split()))
            
            # Candidato si el 70% de las muestras tienen 4+ palabras
            if (word_counts >= 4).mean() > 0.7:
                avg_tokens = word_counts.mean()
                text_candidates[col] = avg_tokens
        except:
            continue
    
    # Seleccionar la mejor columna candidata
    if not text_candidates:
        raise ValueError("No se encontró una columna de texto válida")
    
    text_col = max(text_candidates, key=text_candidates.get)

    # Filtrar y limpiar datos
    df = df[[text_col, class_col]].dropna().copy()
    
    textos_crudos = df[text_col].astype(str).tolist()
    textos_limpios = [limpiar_texto(t) for t in textos_crudos]
    etiquetas = df[class_col].astype(str).tolist()

    return textos_crudos, textos_limpios, etiquetas, text_col, class_col
