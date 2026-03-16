import os
import cv2
import numpy as np
from pathlib import Path

def leer_imagen(ruta):
    """Lee imágenes en rutas con tildes o caracteres especiales en Windows"""
    return cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)

def guardar_imagen(ruta, imagen):
    """Guarda imágenes en rutas con tildes o caracteres especiales en Windows"""
    cv2.imencode('.png', imagen)[1].tofile(ruta)

def get_lab_stats(image_bgr):
    """Calcula la media y desviación estándar en el espacio de color LAB"""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    l_mean, l_std = l.mean(), l.std()
    a_mean, a_std = a.mean(), a.std()
    b_mean, b_std = b.mean(), b.std()
    
    return (l_mean, a_mean, b_mean), (l_std, a_std, b_std)

def apply_reinhard(source_bgr, target_means, target_stds):
    """Aplica la fórmula matemática de Reinhard a una imagen"""
    source_means, source_stds = get_lab_stats(source_bgr)
    
    lab = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    
    l = ((l - source_means[0]) * (target_stds[0] / (source_stds[0] + 1e-5))) + target_means[0]
    a = ((a - source_means[1]) * (target_stds[1] / (source_stds[1] + 1e-5))) + target_means[1]
    b = ((b - source_means[2]) * (target_stds[2] / (source_stds[2] + 1e-5))) + target_means[2]
    
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)
    
    lab_normalized = cv2.merge((l, a, b)).astype(np.uint8)
    return cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)

def process_folder(input_dir, output_dir, template_path):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Cargando imagen de referencia (Template)...")
    template_img = leer_imagen(template_path)
    if template_img is None:
        print(f"❌ Error: No se pudo cargar el template en la ruta: {template_path}")
        return
        
    target_means, target_stds = get_lab_stats(template_img)
    print(f"✅ Template analizado correctamente.\n")
    
    # =========================================================================
    # EL FILTRO DE SEGURIDAD: Solo aceptamos .png (los recortes pequeños)
    # Ignoramos .tif y .svs para evitar que OpenCV estalle.
    # =========================================================================
    valid_extensions = {'.png', '.jpg', '.jpeg'}
    archivos = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    if not archivos:
        print(f"⚠️ No se encontraron recortes (.png o .jpg) en: {input_dir}")
        print("Asegúrate de apuntar a la carpeta donde guardaste los recortes del paso anterior.")
        return
        
    print(f"Iniciando normalización de {len(archivos)} recortes (tiles)...")
    
    procesadas = 0
    for file_name in archivos:
        source_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name)
        
        img = leer_imagen(source_path)
        if img is None:
            continue
            
        norm_img = apply_reinhard(img, target_means, target_stds)
        
        guardar_imagen(output_path, norm_img)
        procesadas += 1
        
        if procesadas % 50 == 0 or procesadas == len(archivos):
            print(f"  Progreso: {procesadas}/{len(archivos)} tiles normalizados...", end='\r')
            
    print(f"\n\n🎉 Proceso completado. {procesadas} imágenes guardadas en: {output_dir}")

if __name__ == "__main__":
    # --- RUTAS ---
    # Asegúrate de que aquí estén LOS RECORTES (los archivos tile_x000_y000.png)
    input_dir = r"D:\Anotaciones\Salidas\BR-007-HYE-25-CONV"
    
    # Aquí se guardarán los recortes normalizados
    output_dir = r"D:\Anotaciones\Recortes_Normalizados"
    
    # Tu imagen de referencia
    template_path = r"D:\Anotaciones\Referencias\BR-007-HYE-25-CONV_tile_x01024_y05632.png"
    
    process_folder(input_dir, output_dir, template_path)