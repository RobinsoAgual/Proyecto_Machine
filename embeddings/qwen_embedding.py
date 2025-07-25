import numpy as np
import requests

def calcular_qwen(textos, api_key):
    """
    Calcula embeddings usando el modelo Qwen a través de la API de DashScope.

    :param textos: Lista de cadenas de texto.
    :param api_key: Tu clave de acceso a la API de DashScope.
    :return: Matriz NumPy con los embeddings generados.
    """
    # URL del endpoint de embeddings (ajústala según el modelo)
    endpoint = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embeddings"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    embeddings = []
    for texto in textos:
        payload = {
            "model": "text-embedding-v2",  # Puedes cambiarlo por otro modelo disponible
            "input": [texto]
        }

        try:
            response = requests.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()

            result = response.json()
            embedding = result['output']['embeddings'][0]['embedding']
            embeddings.append(embedding)

        except Exception as e:
            print(f"⚠️ Error en Qwen embedding para '{texto[:20]}...': {e}")
            # Si hay un error, agregar un vector de ceros como placeholder
            embeddings.append(np.zeros(768))  # Ajustar si es necesario según el modelo

    return np.array(embeddings)