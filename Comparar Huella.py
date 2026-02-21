"""
Sistema de Identificación Biométrica con ORB
Curso: Programación Avanzada

Implementa:
- Preprocesamiento (escala de grises + CLAHE)
- Extracción de características con ORB
- Filtrado con Lowe's Ratio Test
- Selección aleatoria de huella
- Búsqueda iterativa en dataset
- Métrica de tiempo y puntaje

Autor: Cristian David Lopez
"""

import cv2
import os
import random
import time


# ===============================
# CONFIGURACIÓN
# ===============================

DATASET_PATH = "Huellas"     # Carpeta con las 80 imágenes
RATIO_THRESHOLD = 0.75       # Hiperparámetro Lowe
N_FEATURES = 1000            # Número de puntos ORB


# ===============================
# PREPROCESAMIENTO
# ===============================

def preprocess(image_path):
    """
    Aplica normalización:
    - Escala de grises
    - Mejora de contraste con CLAHE
    """

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # CLAHE mejora contraste local sin destruir detalles
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    return enhanced


# ===============================
# EXTRACCIÓN DE CARACTERÍSTICAS
# ===============================

orb = cv2.ORB_create(nfeatures=N_FEATURES)

def extract_features(image):
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


# ===============================
# LOWE'S RATIO TEST
# ===============================

def match_features(des1, des2):
    """
    Aplica Lowe's Ratio Test para filtrar coincidencias falsas
    """

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []

    for m, n in matches:
        if m.distance < RATIO_THRESHOLD * n.distance:
            good_matches.append(m)

    return len(good_matches)


# ===============================
# PROGRAMA PRINCIPAL
# ===============================

def main():

    start_time = time.time()

    images = [img for img in os.listdir(DATASET_PATH)
              if img.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))]

    print("Total imágenes en dataset:", len(images))

    # -----------------------------
    # Selección aleatoria
    # -----------------------------
    unknown_image_name = random.choice(images)
    unknown_image_path = os.path.join(DATASET_PATH, unknown_image_name)

    print("\nHuella seleccionada como desconocida:", unknown_image_name)

    img_unknown = preprocess(unknown_image_path)
    kp1, des1 = extract_features(img_unknown)

    if des1 is None:
        print("No se pudieron extraer características de la huella desconocida.")
        return

    best_score = 0
    best_match = None

    # -----------------------------
    # Búsqueda iterativa
    # -----------------------------
    for img_name in images:

        if img_name == unknown_image_name:
            continue

        img_path = os.path.join(DATASET_PATH, img_name)

        img = preprocess(img_path)
        kp2, des2 = extract_features(img)

        if des2 is None:
            continue

        score = match_features(des1, des2)

        if score > best_score:
            best_score = score
            best_match = img_name

    end_time = time.time()

    # -----------------------------
    # Identificación de clase
    # -----------------------------
    if best_match:
        predicted_user = best_match.split("_")[0]
        real_user = unknown_image_name.split("_")[0]

        print("\n===== RESULTADO =====")
        print("Usuario real:", real_user)
        print("Usuario identificado:", predicted_user)
        print("Puntaje de coincidencia:", best_score)

        if predicted_user == real_user:
            print("Resultado: IDENTIFICACIÓN CORRECTA ✅")
        else:
            print("Resultado: IDENTIFICACIÓN INCORRECTA ❌")

    else:
        print("No se encontró coincidencia suficiente.")

    print("Tiempo de ejecución:",
          round(end_time - start_time, 4), "segundos")


# ===============================
# EJECUCIÓN
# ===============================

if __name__ == "__main__":
    main()