# Procesamiento de Texto con OpenML para Clasificación y Clustering

## Descripción General del Proyecto

Este proyecto implementa un **procesamiento y análisis de datos de texto no estructurado**, enfocado en datasets obtenidos dinámicamente desde la plataforma **OpenML**. El objetivo es facilitar tareas de clasificación y clustering sobre estos textos, aplicando técnicas de procesamiento de lenguaje natural (NLP) y aprendizaje automático de manera automática.

La solución puede recuperar conjuntos de datos directamente desde **OpenML**, limpiar y vectorizar sus columnas de texto, y posteriormente entrenar modelos de clasificación supervisada o realizar agrupamiento (clustering) no supervisado de las instancias.

En particular, el proyecto explora **algoritmos de clustering con restricciones de tamaño de cluster**, que permiten agrupar los datos en clústeres de tamaños predefinidos o balanceados, algo útil para escenarios que requieren grupos equilibrados.

El pipeline está orientado a **usuarios con conocimientos en ciencia de datos o aprendizaje automático** que deseen experimentar con diferentes embeddings de texto (como **TF-IDF** o **Word2Vec**) y algoritmos de clustering avanzados, sin tener que recolectar ni preprocesar manualmente los datos. Además, se provee una **interfaz interactiva** (basada en **Streamlit**) para facilitar la ejecución del pipeline y la visualización de resultados, aunque también es posible usar los componentes del pipeline programáticamente en scripts o notebooks.

## Objetivos del Proyecto

Los objetivos principales de este proyecto son:

- **Recuperación automática de datasets de texto desde OpenML**: 
    Utilizar la API de OpenML para buscar y obtener datasets que contengan texto no estructurado (por ejemplo, reseñas, comentarios, descripciones) junto con etiquetas de clase. Esto permite reunir datos de diversa índole de manera dinámica y sin almacenamiento local fijo.

- **Procesamiento y limpieza del texto**: 
    Aplicar técnicas de preprocesamiento NLP para limpiar los campos de texto, incluyendo eliminación de caracteres especiales, normalización de palabras (minúsculas, acentos), eliminación de stopwords y tokenización. Garantizar que el texto quede en una forma adecuada para ser convertido a vectores numéricos.

- **Vectorización mediante embeddings**:
    Transformar el texto preprocesado en representaciones numéricas (vectores) que puedan ser utilizadas por algoritmos de ML. Se implementan dos enfoques principales de embedding:
    - **TF-IDF**: Representación de texto basada en frecuencias de términos (palabras) ponderadas inversamente por su frecuencia en los documentos (corpus).
    - **Word2Vec**: Representación densa de palabras entrenada mediante modelos de contexto (CBOW o Skip-gram) para capturar similitudes semánticas.

- **Clasificación supervisada (análisis de clases)**:
    Enfocarse en datasets que sean de clasificación, utilizando la columna de clase para evaluar la calidad de las representaciones y proveer un punto de comparación.

- **Clustering con restricciones de tamaño**:
    Agrupar las instancias de texto en clústeres de forma no supervisada, imponiendo restricciones en el tamaño de cada clúster. El proyecto busca formar **clusters balanceados** o con números de elementos predefinidos.

- **Visualización y análisis de resultados**:
    Proporcionar visualizaciones que ayuden a interpretar los resultados, tales como gráficos de dispersión 2D (usando reducción de dimensionalidad PCA) para observar la separación de clusters, gráficos de coeficiente de silueta para evaluar la cohesión/separación de los grupos, y matrices de confusión que comparan los clusters encontrados con las clases originales.

## Flujo de Trabajo del Pipeline

### Etapas del pipeline:

1. **Acceso en caliente a OpenML**:
    El pipeline se conecta a la API de OpenML para obtener datasets directamente desde el repositorio en tiempo real, sin necesidad de descargar manualmente archivos ni almacenarlos localmente. Esto se realiza mediante consultas programáticas a OpenML para listar y cargar datasets.

2. **Extracción de datasets mixtos (texto + clase)**:
    Se filtran los datasets que contienen columnas de texto no estructurado y una columna de etiqueta de clase asociada. Los datasets deben cumplir ciertos criterios, como un número mínimo de instancias.

3. **Limpieza y preprocesamiento del texto**:
    El texto es limpiado mediante eliminación de ruido, conversión a minúsculas, eliminación de stopwords, tokenización, y opcionalmente, lematización.

4. **Generación de embeddings (vectorización)**:
    El texto procesado se convierte en vectores numéricos mediante técnicas de vectorización como **TF-IDF** y **Word2Vec**.

5. **Clasificación por clases (análisis supervisado)**:
    El pipeline permite entrenar modelos de clasificación supervisada usando los embeddings generados para evaluar la calidad de las representaciones.

6. **Clustering con restricciones de tamaño**:
    El algoritmo de clustering es ejecutado sobre los vectores de texto para agrupar las instancias en clústeres con restricciones de tamaño, utilizando enfoques como **K-Means balanceado**, **K-Medoids**, y **MILP**.

## Tecnologías y Bibliotecas Utilizadas

El proyecto hace uso de diversas bibliotecas de Python y herramientas de ciencia de datos para lograr su cometido:

- **OpenML API**: Búsqueda y descarga de datasets desde OpenML.
- **Pandas y NumPy**: Manipulación de datos tabulares y operaciones vectoriales.
- **scikit-learn**: Implementación de algoritmos de ML clásicos, vectorización TF-IDF, y reducción de dimensionalidad.
- **NLTK y langdetect**: Herramientas para el procesamiento de lenguaje natural.
- **Gensim**: Para modelado de tópicos y embeddings de palabras.
- **Sentence Transformers**: Modelos pre-entrenados para generar embeddings de oraciones.
- **OpenAI API, Google Generative AI, Ollama**: Servicios de IA generativa para obtener embeddings pre-entrenados.
- **PuLP (Python Linear Programming)**: Librería para resolver problemas de optimización lineal.
- **PySwarm (PSO)**: Implementación del algoritmo de optimización de enjambre de partículas para clustering.
- **Matplotlib, Seaborn y Plotly**: Bibliotecas para la visualización de resultados.
- **Streamlit**: Framework para la creación de interfaces web interactivas.
- 
## Consideraciones de Ingeniería y Diseño
Durante el desarrollo de este proyecto se tuvieron en cuenta varias decisiones de ingeniería importantes para optimizar su funcionamiento y mantener la coherencia con los objetivos propuestos:
-	No almacenar datasets localmente: Siguiendo una filosofía de datos en caliente, en ningún momento se guardan permanentemente los datasets descargados en disco (ni en Google Drive ni en archivos locales). Cada vez que se necesita un dataset, se descarga mediante la API de OpenML y se mantiene en memoria durante la sesión. Esto asegura trabajar siempre con datos actualizados y evita la proliferación de archivos temporales. Si bien esto implica descargar repetidamente los datos cuando se re-ejecuta el pipeline, los beneficios en limpieza y actualización superan este costo (además OpenML puede cachear datos en la sesión actual, haciendo las cargas subsecuentes más rápidas).
-	Selección automática de columnas de texto y clase: El sistema está diseñado para identificar correctamente, dentro de cada dataset, cuál columna contiene texto libre y cuál es la columna de clase (objetivo). Para ello, se aprovecha la metadata de OpenML (que a veces etiqueta la variable objetivo) y se analizan los tipos de datos de cada columna. Las columnas tipo string o con texto largo son candidatas a ser texto no estructurado, mientras que la columna de clase suele ser categórica con un número limitado de valores únicos. El pipeline verifica que haya al menos una columna de texto y una de clase; de no ser así, el dataset se descarta de la lista de útiles. Este filtrado automático agiliza el flujo, evitando al usuario tener que inspeccionar manualmente cada dataset.
-	Uso de la API de OpenML directamente: En lugar de depender de descargas manuales o de datasets estáticos incluidos en el repositorio, se optó por usar directamente la API REST de OpenML y su paquete oficial en Python. Esto no solo garantiza disponer de muchos más datasets a voluntad, sino que también facilita la reproducibilidad: cualquier usuario en cualquier ubicación puede obtener exactamente los mismos datos a partir de su ID de OpenML. También se evita versionar datos en Git (lo cual es una mala práctica cuando son voluminosos) y se confía en la infraestructura de OpenML para el hospedaje de los mismos.
-	Paralelismo en búsquedas: Para mejorar los tiempos al buscar datasets candidatos en OpenML, se implementó un esquema de búsqueda concurrente. Por ejemplo, al evaluar decenas de posibles datasets (chequeando cuántas instancias tienen, tipos de columnas, etc.), se lanzan peticiones en paralelo utilizando Python threads o asyncio. De esta forma, el filtrado de datasets útiles se acelera significativamente. Esta consideración de eficiencia ayuda porque la base de datos de OpenML es grande y las consultas secuenciales podrían tardar demasiado.
-	Clusters con tamaños personalizados: La principal novedad del proyecto, los algoritmos de clustering con restricciones, requirió decisiones en su implementación. Algunos métodos exactos como MILP garantizan solución óptima pero pueden escalar pobremente con muchos datos (debido a la complejidad combinatoria), por lo que se decidió incluir también métodos heurísticos (PSO, ACO, etc.) que, si bien no garantizan optimalidad, encuentran buenas soluciones en tiempo razonable para conjuntos de datos de tamaño moderado. Asimismo, se mantuvo la posibilidad de usar K-Means estándar como comparación base. Esta combinación de enfoques le da flexibilidad al pipeline para distintos escenarios y tamaños de dataset.
-	No dependencia de servicios externos por defecto: Si bien se integraron opciones para usar embeddings de APIs externas (como OpenAI), por defecto el pipeline funciona completamente offline con opciones open-source (TF-IDF, Word2Vec, sentence transformers locales). De este modo, se evita obligar al usuario a tener claves de API o conexión permanente a servicios de pago para las funcionalidades principales. Las integraciones están ahí solo como complementos.
-	Reproducibilidad y aleatoriedad controlada: Muchos algoritmos aquí (Word2Vec entrenamiento, K-Means inicialización, PSO, etc.) involucran aleatoriedad. Se ha previsto la posibilidad de fijar semillas aleatorias en la interfaz o configuración para obtener resultados reproducibles cuando sea necesario. Esto es importante en un contexto experimental, para poder comparar resultados justos entre corridas.
-	Evitar suposiciones sobre idioma o dominio: Dado que OpenML contiene datasets en múltiples idiomas y dominios (ej. reseñas de productos, noticias, datos médicos, etc.), el pipeline trata de ser lo más genérico posible en el preprocesamiento. Por ejemplo, detecta el idioma del texto para aplicar la lista de stopwords correcta; la tokenización se hace con métodos generales. Aun así, se advierte que la calidad de las técnicas como Word2Vec o TF-IDF dependerá de la cantidad de datos y su homogeneidad. En algunos casos, puede convenir ajustar manualmente el pipeline a un idioma específico (por ejemplo, habilitando lematización solo para inglés o español según corresponda).
-	Documentación y mantenibilidad: El código está estructurado en módulos (algoritmos, embeddings, utils, etc.) para facilitar su mantenimiento y expansión. Cada archivo Python contiene funciones bien definidas para una tarea concreta. Además, este README actúa como guía detallada para nuevos desarrolladores o colaboradores que deseen entender o mejorar alguna parte del pipeline.
Estas consideraciones aseguran que el proyecto se mantenga robusto, flexible y fácil de usar/extender. También reflejan las lecciones aprendidas durante la implementación, como la importancia de no sobrecargar almacenamiento, de apoyarse en APIs estándar, y de proveer suficientes opciones para adaptarse a distintos tipos de datos y requerimientos del usuario.
## Requisitos de Instalación y Entorno

1. **Python**: Versión 3.8 o superior.
2. **Visual Studio Code**: recomendable para ejecutar el proyecto

-	Dependencias principales: Las bibliotecas mencionadas en la sección anterior deben instalarse. Si el proyecto incluye un archivo requirements.txt, puede instalarlas de forma automática con pip. En caso contrario, instálelas manualmente con pip o conda. A continuación se listan los paquetes necesarios:
-	openml (API de OpenML para Python)
-	pandas y numpy
-	scikit-learn
-	nltk
-	gensim
-	sentence-transformers (opcional, para embeddings avanzados)
-	openai, google-generativeai, ollama (opcionales, para embeddings vía APIs externas)
-	langdetect
-	matplotlib, seaborn, plotly
-	tqdm
-	requests
-	pulp (solver MILP)
-	pyswarm (PSO)
-	streamlit (para la interfaz web interactiva)
##	Otros requisitos:
-	Conexión a Internet activa, ya que los datasets se descargan desde OpenML en tiempo real y, si se usan embeddings externos (OpenAI, etc.), también se requieren llamadas a API online.
-	Para el uso de ciertos embeddings (p. ej. OpenAI, Google), necesitará proporcionar credenciales o claves de API válidas y posiblemente configurar variables de entorno para ellas.
-	NLTK: tras instalar, es posible que deba ejecutar una descarga de recursos (nltk.download('stopwords')) para obtener las listas de stopwords, y similarmente asegurarse de tener el soporte para tokenización en el idioma deseado.
-	Si planea usar Sentence Transformers, la primera vez que solicite un modelo (ej: sentence-transformers/all-MiniLM-L6-v2), este será descargado automáticamente. Asegúrese de tener conexión y espacio en disco para almacenarlo.
3. **Dependencias**: Las bibliotecas mencionadas anteriormente deben instalarse. Puedes instalarlas ejecutando:
   ```bash
   pip install -r requirements.txt


## Después de instalar requirements.txt, ejecuta el siguiente comando para descargar el modelo de idioma:

Modelo spaCy:
Después de instalar spaCy, ejecuta el siguiente comando para descargar el modelo de idioma:
   ```bash
python -m spacy download en_core_web_sm

  ```
##Instrucciones de Ejecución del Pipeline
Uso de la interfaz web interactiva (Streamlit):
## 1.- Ejecuta el comando para lanzar la aplicación Streamlit:

 ```bash
streamlit run main.py

  ```
## 2.-Interfaz:

-  La interfaz de Streamlit te permitirá seleccionar un dataset desde OpenML.

- Podrás elegir el método de embedding y configurar el algoritmo de clustering deseado.

-  Visualiza los resultados, como gráficos interactivos y matrices de confusión.

## Ejecución mediante scripts o en entorno de programación:
-  Importa las funciones relevantes desde los módulos del proyecto.

-  Procesa datasets y entrena modelos de clasificación o clustering de forma programática.

## Consideraciones de Ingeniería y Diseño
-  No almacenar datasets localmente: Los datasets se descargan desde OpenML en tiempo real.

-  Selección automática de columnas de texto y clase: Se filtran los datasets adecuados según el tipo de datos.

- Paralelismo en búsquedas: Se emplean técnicas de búsqueda concurrente para acelerar el proceso.

- Clustering con restricciones de tamaño: El pipeline permite especificar tamaños de clusters mediante algoritmos como K-Means balanceado y K-Medoids.

- Reproducibilidad y aleatoriedad controlada: El uso de semillas aleatorias garantiza resultados reproducibles.


## Estructura del Repositorio
 ```bash
Proyecto_Machine/
├── main.py                   # Script principal que lanza la aplicación Streamlit y coordina el pipeline.
├── README.md                 # Documentación detallada (este archivo).
├── algoritmos/               # Implementaciones de algoritmos de clustering con diversas técnicas (balanceo de tamaños).
│   ├── kmeans_ba.py          # Algoritmo K-Means "Best of Seeds" balanceado (busca mejor resultado de K-Means con distintas inicializaciones).
│   ├── kmedoids.py           # Algoritmo de clustering K-Medoids (usa medoides en lugar de centroides, adecuado para ciertos casos).
│   ├── km_milp.py            # Clustering mediante una formulación de K-Means con restricciones de tamaño usando MILP (optimización exacta).
│   ├── hcakc.py              # Variación de clustering jerárquico aglomerativo con control de número de clusters (AKC heuristic).
│   ├── pso.py                # Clustering usando Particle Swarm Optimization (PSO) para distribuir puntos en clusters de tamaño dado.
│   ├── aco.py                # Clustering usando Ant Colony Optimization (ACO) con feromonas para asignar puntos a clusters equilibrados.
│   ├── csclp.py              # Algoritmo de "Constrained Sizes Clustering Problem" (otra metaheurística o método específico para clusters con tamaño).
│   └── sck1.py               # (Posiblemente un algoritmo experimental o una variante especial, incluido para pruebas).
├── embeddings/               # Métodos para generar embeddings (vectores) a partir de texto usando distintas técnicas.
│   ├── tfidf_embedding.py    # Funciones para calcular vectores TF-IDF a partir de una columna de texto.
│   ├── word2vec_embedding.py # Funciones para entrenar un modelo Word2Vec y obtener embeddings promedio de documentos.
│   ├── deepseek_embedding.py # (Integración con un servicio/modelo *DeepSeek* para embeddings - opcional).
│   ├── gemini_embedding.py   # (Integración con modelo *Gemini* - opcional).
│   ├── qwen_embedding.py     # (Integración con modelo *Qwen* - opcional).
│   └── ...                   # (Otros métodos de embedding podrían añadirse aquí, p. ej. chatGPT/OpenAI embeddings).
├── utils/                    # Funciones utilitarias de soporte para carga y preprocesamiento.
│   ├── buscador_dataset_openml.py  # Funciones para listar y filtrar datasets de OpenML según criterios (número de instancias, tipo de datos, etc.).
│   ├── carga_openml.py       # Función para cargar un dataset específico por ID usando la API de OpenML y prepararlo (incluyendo detección de col. de texto/clase).
│   └── preprocesamiento_texto.py   # Funciones de limpieza de texto (eliminar signos, stopwords) y tokenización de cadenas de texto.
├── requirements.txt          # Lista de dependencias Python necesarias para ejecutar el proyecto.
  ```

En la estructura anterior, los nombres entre paréntesis indican módulos opcionales o notas importantes. Por ejemplo, la carpeta datasets/ no existe porque no almacenamos datos localmente, pero se menciona para recalcar ese diseño. Los archivos de algoritmos y embeddings están desacoplados para permitir agregar fácilmente nuevos métodos de clustering o nuevas formas de vectorización sin tener que modificar el núcleo del programa (basta con crear un nuevo .py en la carpeta correspondiente y conectarlo en la interfaz si se desea).


## Conclusión:
- Este proyecto proporciona un enfoque integrado para experimentar con datasets de texto no estructurado desde OpenML, aplicando técnicas de procesamiento de lenguaje natural, clasificación y clustering avanzado con restricciones. A través de una combinación de herramientas open-source y algoritmos personalizados, el pipeline logra automatizar gran parte del trabajo pesado de preparar datos y probar múltiples métodos de agrupamiento. El usuario puede concentrarse en analizar resultados y extraer conclusiones sobre la separación de clases, la calidad de los embeddings o el comportamiento de los algoritmos de clustering.
- Se invita a los usuarios y desarrolladores a explorar el código, probar diferentes configuraciones (por ejemplo, comparar TF-IDF vs Word2Vec, o K-Means vs algoritmos evolutivos) y adaptar el pipeline a sus propias necesidades. Dado que la arquitectura es modular, es relativamente sencillo sustituir componentes (por ejemplo, usar una librería distinta de NLP, o integrar un nuevo algoritmo de clustering que se haya encontrado en la literatura).
- Esperamos que este proyecto sea útil como herramienta educativa y de investigación para entender mejor cómo manejar textos provenientes de múltiples fuentes, cómo evaluar la separabilidad de clases en espacios embebidos, y cómo imponer restricciones adicionales en problemas de clustering. Las aportaciones y feedback son bienvenidos a través del repositorio de GitHub, ya sea en forma de issues, pull requests o discusiones. ¡Que disfrutes explorando y agrupando datos con este pipeline! [1]
## Fuentes:
- [1] How to collect datasets from OpenML using Python | by Young Ben | Medium
https://youngandbin.medium.com/how-to-collect-datasets-from-openml-using-python-3794829bbf5f
- [2] A Gentle Introduction to Word Embedding and Text Vectorization
https://machinelearningmastery.com/a-gentle-introduction-to-word-embedding-and-text-vectorization/
- [3] Balanced clustering - Wikipedia
https://en.wikipedia.org/wiki/Balanced_clustering
