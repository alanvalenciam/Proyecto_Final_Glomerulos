#!/usr/bin/env python3
"""
glomeruli_summary.py

Genera un CSV con el conteo de glomérulos por clase para cada biopsia.

Entrada: Archivos GeoJSON en Entradas/
Salida: CSV en Salidas/glomeruli_summary.csv

Columnas del CSV:
  - Biopsia: nombre del archivo sin extensión
  - Total_Glomerulos: total de glomérulos en la biopsia
  - Proliferativo: conteo de "proliferativo"
  - No_Proliferativo: conteo de "no proliferativo"
  - Esclerosado: conteo de "esclerosado"
  - Excluyente: conteo de "exclude"
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

# Importar funciones reutilizables de tamanio.py
from tamanio import load_geojson, find_geojson_files

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def count_glomeruli_by_class(geojson_path: str) -> Dict[str, int]:
    """
    Cuenta glomérulos por clase en un archivo GeoJSON.

    Args:
        geojson_path: ruta al archivo GeoJSON

    Returns:
        Dict con estructura:
        {
            'total': int,
            'proliferativo': int,
            'no_proliferativo': int,
            'esclerosado': int,
            'excluyente': int,
            'otras': int
        }
    """
    counts = {
        'total': 0,
        'proliferativo': 0,
        'no_proliferativo': 0,
        'esclerosado': 0,
        'excluyente': 0
    }

    try:
        geojson_obj = load_geojson(geojson_path)
        features = geojson_obj.get('features', [])

        for feature in features:
            props = feature.get('properties', {})

            # Extraer clase del GeoJSON
            classification = props.get('classification', {})
            glomeruli_class = classification.get('name', 'Unknown') if isinstance(classification, dict) else 'Unknown'

            # Normalizar nombre de clase (convertir a minúsculas para comparación)
            glomeruli_class_lower = glomeruli_class.lower().strip()

            counts['total'] += 1

            # Mapear clases
            if ('no' in glomeruli_class_lower and 'prolif' in glomeruli_class_lower):
                counts['no_proliferativo'] += 1
            elif 'proliferativo' in glomeruli_class_lower:
                counts['proliferativo'] += 1
            elif 'esclerosado' in glomeruli_class_lower:
                counts['esclerosado'] += 1
            elif 'exclude' in glomeruli_class_lower:
                counts['excluyente'] += 1

    except Exception as e:
        logger.error(f"Error procesando {Path(geojson_path).name}: {e}")

    return counts


def generate_summary_csv(input_dir: str, output_file: str) -> pd.DataFrame:
    """
    Genera CSV con resumen de glomérulos por biopsia.

    Args:
        input_dir: directorio con archivos GeoJSON
        output_file: ruta para guardar el CSV

    Returns:
        DataFrame con el resumen
    """
    logger.info("=" * 70)
    logger.info("GLOMERULI SUMMARY: Conteo de glomérulos por clase")
    logger.info("=" * 70)

    # Buscar archivos GeoJSON
    logger.info(f"\n[1] Buscando archivos GeoJSON en {input_dir}...")
    geojson_files = find_geojson_files(input_dir)

    if not geojson_files:
        logger.error("No se encontraron archivos GeoJSON.")
        return pd.DataFrame()

    logger.info(f"Encontrados {len(geojson_files)} archivos")

    # Procesar cada biopsia
    logger.info(f"\n[2] Procesando {len(geojson_files)} biopsias...")
    rows = []

    for idx, geojson_path in enumerate(geojson_files, 1):
        biopsia_name = Path(geojson_path).stem
        counts = count_glomeruli_by_class(geojson_path)

        row = {
            'Biopsia': biopsia_name,
            'Total_Glomerulos': counts['total'],
            'Proliferativo': counts['proliferativo'],
            'No_Proliferativo': counts['no_proliferativo'],
            'Esclerosado': counts['esclerosado'],
            'Excluyente': counts['excluyente']
        }
        rows.append(row)

        logger.info(
            f"  [{idx}/{len(geojson_files)}] {biopsia_name}: "
            f"Total={counts['total']}, "
            f"Prol={counts['proliferativo']}, "
            f"NoProl={counts['no_proliferativo']}, "
            f"Escl={counts['esclerosado']}, "
            f"Excl={counts['excluyente']}"
        )

    # Crear DataFrame
    df = pd.DataFrame(rows)

    # Guardar CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"\n[3] CSV guardado: {output_file}")

    # Mostrar estadísticas generales
    logger.info("\n" + "=" * 70)
    logger.info("ESTADÍSTICAS GENERALES")
    logger.info("=" * 70)
    logger.info(f"Total de biopsias: {len(df)}")
    logger.info(f"Total de glomérulos: {df['Total_Glomerulos'].sum()}")
    logger.info("")
    logger.info("RESUMEN POR CLASE:")
    logger.info(f"  Proliferativo:      {df['Proliferativo'].sum():>6} ({100*df['Proliferativo'].sum()/df['Total_Glomerulos'].sum():>5.1f}%)")
    logger.info(f"  No Proliferativo:   {df['No_Proliferativo'].sum():>6} ({100*df['No_Proliferativo'].sum()/df['Total_Glomerulos'].sum():>5.1f}%)")
    logger.info(f"  Esclerosado:        {df['Esclerosado'].sum():>6} ({100*df['Esclerosado'].sum()/df['Total_Glomerulos'].sum():>5.1f}%)")
    logger.info(f"  Excluyente:         {df['Excluyente'].sum():>6} ({100*df['Excluyente'].sum()/df['Total_Glomerulos'].sum():>5.1f}%)")
    logger.info("=" * 70)

    # Mostrar primeras filas
    logger.info("\nPRIMERAS FILAS DEL CSV:")
    logger.info(df.head(10).to_string(index=False))
    logger.info("")

    return df


if __name__ == "__main__":
    INPUT_DIR = "/Users/olivera/Documents/Proyecto_Final_Glomerulos/Entradas"
    OUTPUT_CSV = "/Users/olivera/Documents/Proyecto_Final_Glomerulos/Salidas/glomeruli_summary.csv"

    df_summary = generate_summary_csv(INPUT_DIR, OUTPUT_CSV)

    if not df_summary.empty:
        logger.info("✓ DONE - Resumen generado exitosamente")
    else:
        logger.error("✗ ERROR - No se pudo generar el resumen")
        exit(1)
