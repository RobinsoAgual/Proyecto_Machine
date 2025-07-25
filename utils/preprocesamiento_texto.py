# utils/preprocesamiento_texto.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# Descargar recursos de NLTK si no están disponibles
nltk.download("stopwords")
nltk.download("punkt")

# Inicializar recursos
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

def limpiar_texto(texto):
    """
    Limpia el texto eliminando caracteres especiales, stopwords y aplicando stemming.
    
    Args:
        texto (str): Texto a limpiar.
    
    Returns:
        str: Texto limpio.
    """
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)  # Eliminar caracteres no alfabéticos
    palabras = texto.split()
    palabras = [stemmer.stem(p) for p in palabras if p not in stop_words]
    return " ".join(palabras)

def tokenizar_texto(texto):
    """
    Tokeniza el texto eliminando puntuación y retornando solo palabras.
    
    Args:
        texto (str): Texto a tokenizar.
    
    Returns:
        list: Lista de tokens.
    """
    return tokenizer.tokenize(texto)