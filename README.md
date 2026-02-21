# Sistema de Identificación Biométrica con ORB

Un sistema de reconocimiento de huellas dactilares implementado con OpenCV que utiliza el algoritmo **ORB (Oriented FAST and Rotated BRIEF)** para la extracción de características y el filtrado de coincidencias con **Lowe's Ratio Test**.

## Características

- ✓ **Preprocesamiento**: Conversión a escala de grises + mejora de contraste con CLAHE
- ✓ **Extracción de características**: Algoritmo ORB con 1000 puntos de características
- ✓ **Filtrado inteligente**: Implementación de Lowe's Ratio Test (umbral: 0.75)
- ✓ **Selección aleatoria**: Elige una huella desconocida del dataset
- ✓ **Búsqueda iterativa**: Compara con todas las huellas del dataset
- ✓ **Identificación de clase**: Extrae el usuario de la huella identificada
- ✓ **Métricas**: Tiempo de ejecución y puntaje de coincidencia

## Requisitos

- Python 3.8+
- OpenCV (`cv2`)

## Instalación

```bash
pip install opencv-python
```

## Uso

```bash
python "Comparar Huella.py"
```

El script:
1. Carga todas las imágenes de huellas desde la carpeta `Huellas/`
2. Selecciona aleatoriamente una huella como "desconocida"
3. Extrae características con ORB
4. Busca iterativamente en todas las otras huellas
5. Identifica al usuario con la mejor coincidencia
6. Muestra el resultado (correcta/incorrecta) y el tiempo de ejecución

## Ejemplo de Salida

```
Total imágenes en dataset: 80

Huella seleccionada como desconocida: 108_6.tif

===== RESULTADO =====
Usuario real: 108
Usuario identificado: 108
Puntaje de coincidencia: 42
Resultado: IDENTIFICACIÓN CORRECTA ✅
Tiempo de ejecución: 2.1394 segundos
```

## Estructura del Proyecto

```
ReconocimientoDeHuella/
├── Comparar Huella.py       # Script principal
├── Huellas/                 # Dataset con 80 imágenes de huellas (8 por usuario, 10 usuarios)
│   ├── 101_1.tif
│   ├── 101_2.tif
│   └── ...
└── README.md               # Este archivo
```

## Configuración

Puedes ajustar los siguientes parámetros en el código:

```python
DATASET_PATH = "Huellas"     # Carpeta con las imágenes
RATIO_THRESHOLD = 0.75       # Umbral para Lowe's Ratio Test
N_FEATURES = 1000            # Número de puntos ORB a detectar
```

## Algoritmos Utilizados

### ORB (Oriented FAST and Rotated BRIEF)
- Detector de esquinas rápido y eficiente
- Descriptor de características robusto a rotaciones
- Ideal para búsqueda de coincidencias en tiempo real

### CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Mejora el contraste local sin aumentar el ruido
- Parámetros: `clipLimit=3.0`, `tileGridSize=(8, 8)`

### Lowe's Ratio Test
- Filtra coincidencias falsas comparando distancias de los dos vecinos más cercanos
- Una coincidencia es válida si: `distancia_mejor < umbral × distancia_segundo_mejor`

### BFMatcher (Brute Force Matcher)
- Compara descriptores utilizando distancia de Hamming
- Apropiado para descriptores binarios como BRIEF

## Dataset

El dataset contiene 80 imágenes de huellas dactilares en formato TIFF:
- **10 usuarios** (101-110)
- **8 muestras por usuario** (variaciones de presión, ángulo, etc.)
- Formato: `[usuario]_[muestra].tif`

## Autor

Cristian David Lopez - Curso Programación Avanzada

## Licencia

Este proyecto es parte de un examen académico.
