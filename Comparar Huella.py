"""
Sistema de Identificación Biométrica con ORB
Curso: Programación Avanzada
Institución Universitaria Pascual Bravo

Implementa:
- Preprocesamiento (escala de grises + CLAHE)
- Extracción de características con ORB (FAST + BRIEF)
- Filtrado con Lowe's Ratio Test
- Selección aleatoria de huella y búsqueda iterativa
- Visualización de puntos clave (Minucias)
- Métrica de tiempo y puntaje

Autor: Cristian David Lopez
Fecha: Febrero de 2026
"""

import cv2
import os
import random
import time
import numpy as np

# ===============================
# CONFIGURACIÓN
# ===============================
DATASET_PATH = "Huellas"     # Carpeta con las 80 imágenes [cite: 55]
RATIO_THRESHOLD = 0.75       # Hiperparámetro de confianza de Lowe [cite: 51]
N_FEATURES = 1000            # Número de puntos clave ORB a detectar [cite: 26]

# ===============================
# PREPROCESAMIENTO
# ===============================
def preprocess(image_path):
    """
    Normaliza la imagen para resaltar crestas y eliminar variaciones 
    de iluminación mediante CLAHE[cite: 45].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    # Paso 1: Escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Paso 2: CLAHE (Mejora de contraste adaptativa)
    # Resuelve el problema de manchas por sudor o presión [cite: 46]
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced

# ===============================
# EXTRACCIÓN DE CARACTERÍSTICAS (ORB)
# ===============================
# ORB utiliza FAST para localizar minucias y BRIEF para descriptores [cite: 35, 36]
orb = cv2.ORB_create(nfeatures=N_FEATURES)

def extract_features(image):
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# ===============================
# LOWE'S RATIO TEST
# ===============================
def match_features(des1, des2):
    """
    Asegura que una coincidencia sea genuina si es significativamente 
    mejor que la segunda opción[cite: 50].
    """
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        # Aplicación del umbral de confianza [cite: 49]
        if m.distance < RATIO_THRESHOLD * n.distance:
            good_matches.append(m)

    return good_matches

# ===============================
# PROGRAMA PRINCIPAL
# ===============================
def main():
    start_time = time.time() # Métrica de desempeño [cite: 60]

    # Carga del dataset [cite: 55]
    if not os.path.exists(DATASET_PATH):
        print(f"Error: No se encuentra la carpeta '{DATASET_PATH}'")
        return

    images = [img for img in os.listdir(DATASET_PATH)
              if img.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))]

    if not images:
        print("Dataset vacío o formato no soportado.")
        return

    # 1. Selección Aleatoria [cite: 57]
    unknown_image_name = random.choice(images)
    unknown_image_path = os.path.join(DATASET_PATH, unknown_image_name)

    print(f"Total imágenes en dataset: {len(images)} [cite: 55]")
    print(f"Huella seleccionada (incógnita): {unknown_image_name} [cite: 57]")

    img_unknown = preprocess(unknown_image_path)
    kp_unk, des_unk = extract_features(img_unknown)

    if des_unk is None:
        print("No se pudieron extraer características de la huella incógnita.")
        return

    best_score = 0
    best_match_name = None
    best_kp = None
    best_des = None
    best_good_matches = []

    # 2. Búsqueda Iterativa [cite: 58]
    for img_name in images:
        if img_name == unknown_image_name:
            continue

        img_path = os.path.join(DATASET_PATH, img_name)
        img_db = preprocess(img_path)
        kp_db, des_db = extract_features(img_db)

        if des_db is None: continue

        # Obtener coincidencias filtradas [cite: 47]
        current_matches = match_features(des_unk, des_db)
        score = len(current_matches)

        if score > best_score:
            best_score = score
            best_match_name = img_name
            best_kp = kp_db
            best_des = des_db
            best_good_matches = current_matches

    end_time = time.time()

    # 3. Identificación de Clase y Resultados [cite: 59, 60]
    if best_match_name:
        predicted_user = best_match_name.split("_")[0]
        real_user = unknown_image_name.split("_")[0]

        print("\n" + "="*25)
        print(f"USUARIO REAL: {real_user}")
        print(f"USUARIO IDENTIFICADO: {predicted_user} [cite: 59]")
        print(f"PUNTAJE DE COINCIDENCIA: {best_score} [cite: 60]")
        print(f"TIEMPO DE EJECUCIÓN: {round(end_time - start_time, 4)} seg [cite: 60]")

        if predicted_user == real_user:
            print("RESULTADO: IDENTIFICACIÓN CORRECTA ✅")
        else:
            print("RESULTADO: IDENTIFICACIÓN INCORRECTA ❌")
        print("="*25)

        # 4. Visualización de puntos de referencia (Minucias)
        # Cargamos la imagen original del mejor match para comparar
        img_best = preprocess(os.path.join(DATASET_PATH, best_match_name))
        
        # Dibujamos las conexiones entre minucias [cite: 13, 35]
        # Se limitan a las 50 mejores para claridad visual
        img_result = cv2.drawMatches(img_unknown, kp_unk, img_best, best_kp, 
                                     best_good_matches[:50], None, 
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow("Comparativa de Minucias - Sistema Biometrico", img_result)
        print("\nMostrando mapa de minucias en pantalla... Presione cualquier tecla para cerrar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("No se encontró coincidencia suficiente en el dataset.")

if __name__ == "__main__":
    main()

"""
======================================================================
6. INFORME TÉCNICO DE EVALUACIÓN (ENTREGABLE 2) [cite: 68]
======================================================================

Pregunta 1: ¿Por qué una huella podría dar un puntaje bajo a pesar de 
ser del mismo usuario? [cite: 69]

* Factores Físicos: El sudor, la presión inconsistente sobre el sensor 
  o la suciedad del dedo alteran la visibilidad de las minucias[cite: 46].
* Ruido de Captura: La luz del escáner puede crear variaciones de 
  iluminación (manchas grises) que dificultan la binarización y el 
  resaltado de crestas[cite: 45, 46].
* Desplazamiento: Si el dedo se coloca en una posición muy parcial, 
  el algoritmo ORB detectará menos puntos clave comunes respecto a la 
  imagen original de la base de datos[cite: 42].

Pregunta 2: ¿Cómo escalaría este sistema si en lugar de 80 imágenes 
tuviera 80,000 (énfasis en computación en la nube)? [cite: 70]

* Plantillas Digitales: No se deben procesar imágenes en tiempo real. 
  Se extraen los descriptores (minucias) una vez y se almacenan como 
  plantillas binarias compactas[cite: 14, 37].
* Búsqueda Vectorial e Indexación: Se implementarían algoritmos de 
  búsqueda de vecinos más cercanos (ej. FAISS) para encontrar matches 
  en milisegundos sin necesidad de un ciclo iterativo manual[cite: 47].
* Procesamiento en la Nube: El uso de clusters de cómputo permitiría 
  la paralelización de tareas, donde múltiples servidores comparan 
  segmentos del dataset simultáneamente para reducir la latencia[cite: 43].
* In-Memory Databases: Para 80,000 registros, las plantillas pueden 
  alojarse en memoria RAM para garantizar una validación de identidad 
  casi instantánea.
======================================================================
"""