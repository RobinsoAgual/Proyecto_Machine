import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils.carga_openml import cargar_dataset_openml

def buscar_datasets_utiles(min_instancias=100, max_instancias=1000, max_features=20, limite=15):
    """
    Busca datasets útiles en OpenML aplicando filtros y validaciones.
    """
    print("🌐 Conectándose a OpenML...")
    
    # Importar openml dentro de la función para manejar errores
    try:
        import openml
    except ImportError as e:
        raise ImportError("OpenML no está instalado. Ejecuta 'pip install openml'") from e

    # Obtener lista de datasets disponibles en OpenML
    all_datasets = openml.datasets.list_datasets(output_format='dataframe')
    print(f"📊 Total de datasets disponibles en OpenML: {len(all_datasets)}")

    print("\n🔍 Aplicando filtros básicos...")
    filtered = all_datasets[
        (all_datasets['NumberOfInstances'] >= min_instancias) &
        (all_datasets['NumberOfInstances'] <= max_instancias) &
        (all_datasets['NumberOfFeatures'] <= max_features) &
        (all_datasets['NumberOfMissingValues'] == 0)
    ].drop_duplicates(subset='did')

    print(f"🔎 {len(filtered)} datasets cumplen con los filtros iniciales.\n")

    # Evaluar datasets en paralelo
    validos = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(evaluar_dataset, did) for did in filtered['did']]
        for future in tqdm(futures, total=len(futures)):
            info = future.result()
            if info:
                validos.append(info)
                if len(validos) >= limite:
                    break

    print(f"\n✅ Total de datasets útiles encontrados: {len(validos)}\n")
    for ds in validos:
        print(f"📦 {ds['name']}")
        print(f"   - Columna de texto: {ds['text_column']}")
        print(f"   - Columna de clase: {ds['class_column']}")
        print(f"   - Instancias: {ds['n_instances']} | Variables: {ds['n_features']}")
        print("------")

    return validos


def evaluar_dataset(did):
    """
    Evalúa un dataset individual para determinar si es útil.
    """
    try:
        # Usar la función cargar_dataset_openml para obtener información del dataset
        _, _, _, text_col, class_col = cargar_dataset_openml(did)

        # Descargar el dataset nuevamente para obtener metadatos adicionales
        import openml
        dataset = openml.datasets.get_dataset(did)
        df, _, _, _ = dataset.get_data()

        # Validar que las columnas detectadas existan
        if text_col not in df.columns or class_col not in df.columns:
            return None

        # Retornar información del dataset
        return {
            "did": did,
            "name": dataset.name,
            "n_instances": df.shape[0],
            "n_features": df.shape[1],
            "text_column": text_col,
            "class_column": class_col
        }
    except Exception as e:
        print(f"⚠️ Dataset {did} falló: {e}")
        return None


if __name__ == "__main__":
    buscar_datasets_utiles()