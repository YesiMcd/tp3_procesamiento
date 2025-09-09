import cv2
import numpy as np
import glob

import sys
import os

# Agrega la carpeta raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


from machine.utils.hu_moments_generation import hu_moments_of_file
from machine.utils.label_converters import int_to_label


def load_and_test(model, test_dir='C:/Users/admin/Desktop/2025-2C/procesamiento/reconocimiento_piezasajedrez/machine/shapes'):
    """
    Prueba un modelo con imágenes de piezas de ajedrez.
    
    Args:
        model: modelo entrenado que tenga método predict().
        test_dir: carpeta con imágenes de prueba, cada imagen individual.
    """
    if not os.path.exists(test_dir):
        raise Exception(f"Test directory does not exist: {test_dir}")

    files = glob.glob(os.path.join(test_dir, '*'))
    if len(files) == 0:
        raise Exception(f"No images found in test directory: {test_dir}")

    for f in files:
        # Genera los momentos de Hu de la imagen
        hu_moments = hu_moments_of_file(f)

        # Prepara la muestra para el modelo
        sample = np.array(hu_moments, dtype=np.float32).reshape(1, -1)
        test_response = model.predict(sample)[0]

        # Lee la imagen y dibuja el label predicho
        image = cv2.imread(f)
        label_text = int_to_label(test_response)
        image_with_text = cv2.putText(
            image,
            label_text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("result", image_with_text)
        cv2.waitKey(0)

    cv2.destroyAllWindows()