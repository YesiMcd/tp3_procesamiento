import cv2
import csv
import glob
import numpy as np
import math
import os

# Carpeta donde están las imágenes de las piezas
DATA_DIR = 'C:/Users/admin/Desktop/2025-2C/procesamiento/reconocimiento_piezasajedrez/machine/shapes'  # cada subcarpeta es una pieza.
OUTPUT_FILE = './generated-files/chess-hu-moments.csv'

# Escribe los valores de los momentos de Hu en el archivo
def write_hu_moments(label, writer):
    path = os.path.join(DATA_DIR, label, '*')
    files = glob.glob(path)
    hu_moments = []

    for file in files:
        hu_moments.append(hu_moments_of_file(file))

    for mom, file in zip(hu_moments, files):
        if mom is not None:
            writer.writerow(mom)

# Calcula los momentos de Hu de un archivo de imagen y características extra
def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2
    )

    # Invertimos la imagen si las piezas son oscuras sobre fondo claro
    bin_img = 255 - bin_img

    # Eliminamos ruido (apertura para no perder detalles pequeños)
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None  # En caso de que no encuentre contornos

    shape_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(shape_contour)
    huMoments = cv2.HuMoments(moments)

    # Escalado logarítmico
    for i in range(7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]) + 1e-30)

    # --- Características extra para diferenciar rey y reina ---
    # Convex hull y defectos
    hull = cv2.convexHull(shape_contour, returnPoints=False)
    if hull is not None and len(hull) > 3:
        defects = cv2.convexityDefects(shape_contour, hull)
        n_defects = 0 if defects is None else defects.shape[0]
    else:
        n_defects = 0

    # Aspect ratio
    x, y, w, h = cv2.boundingRect(shape_contour)
    aspect_ratio = float(w) / h

    # Solidity
    area = cv2.contourArea(shape_contour)
    hull_points = cv2.convexHull(shape_contour)
    hull_area = cv2.contourArea(hull_points)
    solidity = 0 if hull_area == 0 else float(area) / hull_area

    # Flatten Hu Moments y agregar características extra + label
    flattened = huMoments.ravel()
    row = np.append(flattened, [n_defects, aspect_ratio, solidity, os.path.basename(os.path.dirname(filename))])

    return row

# Genera el archivo CSV con todos los momentos de Hu
def generate_hu_moments_file():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        # Lista de piezas de ajedrez (nombres de carpetas)
        pieces = ['peon', 'torre', 'caballo', 'alfil', 'reina', 'rey']
        for piece in pieces:
            write_hu_moments(piece, writer)
    print(f"Archivo generado en {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_hu_moments_file()
