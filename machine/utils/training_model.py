import csv
import numpy as np
from machine.utils.label_converters import label_to_int
from sklearn.tree import DecisionTreeClassifier
import os

# ----------------------------------------
# Carga el dataset de entrenamiento desde el CSV generado
# ----------------------------------------
def load_training_set():
    csv_file_path = './generated-files/chess-hu-moments.csv'

    if not os.path.exists(csv_file_path):
        raise Exception(f"CSV file not found: {csv_file_path}. Asegurate de generar los Hu Moments primero.")

    train_data = []
    train_labels = []

    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop()  # último elemento = label
            floats = [float(n) for n in row]  # momentos de Hu
            train_data.append(np.array(floats, dtype=np.float32))
            train_labels.append(np.array([label_to_int(class_label)], dtype=np.int32))

    train_data = np.array(train_data, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)

    return train_data, train_labels

# ----------------------------------------
# Entrena un modelo Decision Tree con los datos
# ----------------------------------------
def train_model():
    train_data, train_labels = load_training_set()

    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(train_data, train_labels.ravel())

    print("Modelo entrenado con éxito.")
    return tree

# ----------------------------------------
if __name__ == "__main__":
    model = train_model()
    print("Modelo listo para usar con load_and_test(model).")
