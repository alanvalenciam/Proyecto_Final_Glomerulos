import os
from PIL import Image
import numpy as np
from pathlib import Path

# Aumentar límite de píxeles para imágenes gigantes
Image.MAX_IMAGE_PIXELS = None 

def process_folder_to_subfolders(input_dir, output_dir, tile_size=512, overlap=0, bg_threshold=0.8):
    # Crea el directorio de salida principal si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Definir qué tipos de archivos vamos a procesar
    valid_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.svs'}
    
    # Obtener lista de archivos válidos en la carpeta de entrada
    archivos = os.listdir(input_dir)
    imagenes_a_procesar = [f for f in archivos if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not imagenes_a_procesar:
        print(f"No se encontraron imágenes en: {input_dir}")
        return

    print(f"Se encontraron {len(imagenes_a_procesar)} imágenes para procesar.\n")
    
    # Recorrer cada imagen de la carpeta
    for file_name in imagenes_a_procesar:
        image_path = os.path.join(input_dir, file_name)
        base_name = os.path.splitext(file_name)[0] # Nombre sin la extensión
        
        # =========================================================
        # NUEVO: Crear una subcarpeta específica para esta imagen
        # =========================================================
        image_output_dir = os.path.join(output_dir, base_name)
        Path(image_output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"▶ Procesando: {file_name}")
        print(f"  Creando subcarpeta: {image_output_dir}")
        
        try:
            img = Image.open(image_path)
            width, height = img.size
            print(f"  Dimensiones: {width}x{height}")
            
            stride = tile_size - overlap
            saved_count = 0
            
            # Recorrer la imagen completa
            for y in range(0, height, stride):
                for x in range(0, width, stride):
                    
                    # Definir coordenadas reales
                    x_end = min(x + tile_size, width)
                    y_end = min(y + tile_size, height)
                    
                    # Extraer el recorte
                    tile = img.crop((x, y, x_end, y_end))
                    
                    # Padding (Relleno) si el tile es más pequeño que 512x512
                    if tile.size != (tile_size, tile_size):
                        new_tile = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
                        new_tile.paste(tile, (0, 0))
                        tile = new_tile
                    
                    # Verificar fondo para descartar tiles vacíos
                    if not is_mostly_background(tile, bg_threshold):
                        # Guardamos el tile DENTRO de la nueva subcarpeta (image_output_dir)
                        tile_name = f"{base_name}_tile_x{x:05d}_y{y:05d}.png"
                        tile_path = os.path.join(image_output_dir, tile_name)
                        tile.save(tile_path)
                        saved_count += 1
                        
                        # Mostrar progreso en la misma línea
                        if saved_count % 100 == 0:
                            print(f"  Guardados {saved_count} tiles...", end='\r')

            print(f"\n  ✅ Completado. Tiles útiles: {saved_count} guardados en su carpeta.\n")
            
        except Exception as e:
            print(f"\n  ❌ Error procesando {file_name}: {e}\n")
        
    print("🎉 ¡Proceso de toda la carpeta finalizado!")

def is_mostly_background(tile, threshold=0.8):
    gray = np.array(tile.convert('L'))
    # 140 es el umbral para considerar un píxel como "blanco/beige claro"
    white_pixels = np.sum(gray > 140) 
    white_ratio = white_pixels / gray.size
    return white_ratio > threshold

if __name__ == "__main__":
    # Carpeta que contiene las imágenes (los .tiff optimizados que generaste)
    input_dir = r"D:\Anotaciones\Entradas"
    
    # Carpeta raíz donde se crearán todas las subcarpetas
    output_dir = r"D:\Anotaciones\Salidas"
    # Ejecutar la función
    process_folder_to_subfolders(input_dir, output_dir)