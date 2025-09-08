import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def label_to_int(string_label):
    if string_label == 'peon': return 1
    if string_label == 'torre': return 2
    if string_label == 'caballo': return 3
    if string_label == 'alfil': return 4
    if string_label == 'reina': return 5
    if string_label == 'rey': return 6
    else:
        raise Exception('Unknown class_label')

def int_to_label(int_label):
    if int_label == 1: return 'peon'
    if int_label == 2: return 'torre'
    if int_label == 3: return 'caballo'
    if int_label == 4: return 'alfil'
    if int_label == 5: return 'reina'
    if int_label == 6: return 'rey'
    else:
        raise Exception('Unknown class_label')

trainData = []
trainLabels = []

def load_training_set():
    global trainData
    global trainLabels
    trainData = []
    trainLabels = []
    with open('./generated-files/chess-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop()
            floats = [float(n) for n in row]
            trainData.append(np.array(floats, dtype=np.float32))
            trainLabels.append(np.array([label_to_int(class_label)], dtype=np.int32))
    trainData = np.array(trainData, dtype=np.float32)
    trainLabels = np.array(trainLabels, dtype=np.int32)

def train_model():
    load_training_set()
    tree = DecisionTreeClassifier(max_depth=10)
    tree.fit(trainData, trainLabels.ravel())
    return tree

