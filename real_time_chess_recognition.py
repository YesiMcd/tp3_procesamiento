import cv2
import numpy as np
import math

import sys
import os

# Añade la carpeta raíz del proyecto al path
#project_root = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(project_root)

# Añade la carpeta 'machines' al path para que Python encuentre 'utils'
sys.path.append(os.path.join(os.path.dirname(__file__), "machine"))

for p in sys.path:
    print(p)

from machine.utils.contour import get_contours, filter_contours_by_area, get_bounding_rect
from utils.frame_editor import apply_color_convertion, threshold, denoise, draw_contours
from trackbar import create_trackbar, get_trackbar_value
from ml_model import train_model
from utils.label_converters import int_to_label

# Colores BGR para cada pieza de ajedrez
COLOR_PAWN = (200, 200, 200)
COLOR_ROOK = (255, 0, 0)
COLOR_KNIGHT = (0, 255, 0)
COLOR_BISHOP = (0, 0, 255)
COLOR_QUEEN = (255, 255, 0)
COLOR_KING = (255, 0, 255)

# Diccionario para asignar color según label
color_map = {
    'peon': COLOR_PAWN,
    'torre': COLOR_ROOK,
    'caballo': COLOR_KNIGHT,
    'alfil': COLOR_BISHOP,
    'reina': COLOR_QUEEN,
    'rey': COLOR_KING
}

def main():
    window_name = 'Chess Recognition'
    cv2.namedWindow(window_name)
    cap = cv2.VideoCapture(0)

    # Entrena o carga tu modelo
    model = train_model()

    # Trackbars para parámetros en tiempo real
    create_trackbar('Threshold', window_name, 255)
    create_trackbar('Min Area', window_name, 100)
    create_trackbar('Max Area', window_name, 5000)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir a gris
        gray_frame = apply_color_convertion(frame=frame, color=cv2.COLOR_BGR2GRAY)

        # Obtener valores de trackbars
        thresh_val = get_trackbar_value('Threshold', window_name)
        min_area = get_trackbar_value('Min Area', window_name)
        max_area = get_trackbar_value('Max Area', window_name)

        # Aplicar umbral y denoise
        thresh_frame = threshold(frame=gray_frame, slider_max=255,
                                 binary=cv2.THRESH_BINARY,
                                 trackbar_value=thresh_val)
        frame_denoised = denoise(frame=thresh_frame, method=cv2.MORPH_ELLIPSE, radius=5)

        # Detectar y filtrar contornos
        contours = get_contours(frame=frame_denoised, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
        filtered_contours = filter_contours_by_area(contours, min_area, max_area)

        for cont in filtered_contours:
            # Calcular Hu Moments
            mom = cv2.moments(cont)
            hu_moments = cv2.HuMoments(mom)
            for i in range(7):
                if hu_moments[i] != 0:
                    hu_moments[i] = -1 * math.copysign(1.0, hu_moments[i]) * math.log10(abs(hu_moments[i]))

            # Predecir pieza
            sample = np.array(hu_moments, dtype=np.float32).reshape(1, -1)
            pred_int = model.predict(sample)[0]
            label = int_to_label(pred_int)

            # Dibuja contorno y escribe el label
            color = color_map.get(label, (0, 255, 255))  # amarillo si no reconoce
            draw_contours(frame=frame, contours=[cont], color=color, thickness=2)
            x, y, _, _ = get_bounding_rect(cont)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Mostrar ventanas
        cv2.imshow(window_name, frame)
        cv2.imshow('Processed', frame_denoised)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
