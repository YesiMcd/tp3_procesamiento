import cv2
import csv
import glob
import numpy as np
import math
import os

# Carpeta donde están las imágenes de las piezas
DATA_DIR = 'C:/Users/admin/Desktop/2025-2C/Procesamiento imagenes y señales/reconocimiento_piezasajedrez/machine/shapes'  # cada subcarpeta es una pieza.
OUTPUT_FILE = './generated-files/chess-hu-moments.csv'

# Escribe los valores de los momentos de Hu en el archivo
def write_hu_moments(label, writer):
    path = os.path.join(DATA_DIR, label, '*')
    files = glob.glob(path)
    hu_moments = []

    for file in files:
        hu_moments.append(hu_moments_of_file(file))

    for mom in hu_moments:
        flattened = mom.ravel()
        row = np.append(flattened, label)
        writer.writerow(row)


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


# Calcula los momentos de Hu de un archivo de imagen
def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bin_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 67, 2
    )

    # Invertimos la imagen si las piezas son oscuras sobre fondo claro
    bin_img = 255 - bin_img

    # Eliminamos ruido
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_ERODE, kernel)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.zeros((7, 1))  # En caso de que no encuentre contornos

    shape_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(shape_contour)
    huMoments = cv2.HuMoments(moments)

    # Escalado logarítmico
    for i in range(7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]) + 1e-30)

    return huMoments


if __name__ == "__main__":
    generate_hu_moments_file()