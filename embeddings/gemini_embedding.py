import numpy as np
import google.generativeai as genai
from utils.preprocesamiento_texto import limpiar_texto

# Configura tu API key de Gemini
genai.configure(api_key="AIzaSyDjO2AfwM3-zaYNL8VH-bm2IgDDq8Dqaio")

def dividir_texto(texto, max_length=1000):
    """
    Divide un texto en fragmentos de longitud máxima especificada.
    """
    palabras = texto.split()
    fragmentos = []
    fragmento_actual = []

    for palabra in palabras:
        if len(" ".join(fragmento_actual + [palabra])) <= max_length:
            fragmento_actual.append(palabra)
        else:
            fragmentos.append(" ".join(fragmento_actual))
            fragmento_actual = [palabra]

    if fragmento_actual:
        fragmentos.append(" ".join(fragmento_actual))

    return fragmentos


def calcular_gemini(textos):
    """
    Calcula embeddings usando el modelo Gemini.
    """
    print("🔷 Calculando embeddings con Gemini...")
    embeddings = []

    for texto in textos:
        try:
            # Limpiar el texto
            texto_limpio = limpiar_texto(texto)

            # Dividir el texto en fragmentos si supera el límite de longitud
            fragmentos = dividir_texto(texto_limpio, max_length=3000)  # Ajusta el límite según Gemini

            # Calcular el embedding promedio de los fragmentos
            embeddings_fragmento = []
            for fragmento in fragmentos:
                response = genai.embed_content(
                    model="models/embedding-001",
                    content=fragmento,
                    task_type="retrieval_document"
                )
                embeddings_fragmento.append(response["embedding"])

            # Calcular el embedding promedio
            embedding_promedio = np.mean(embeddings_fragmento, axis=0)
            embeddings.append(embedding_promedio)

        except Exception as e:
            print(f"⚠️ Error en Gemini embedding para '{texto[:20]}...': {e}")
            embeddings.append(np.zeros(768))  # Embedding de respaldo (vector de ceros)

    return np.array(embeddings)
