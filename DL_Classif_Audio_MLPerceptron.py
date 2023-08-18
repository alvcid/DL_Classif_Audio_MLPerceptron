import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pydub import AudioSegment
from pydub.playback import play
import librosa
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, confusion_matrix

#Definimos un conjunto de constantes con las rutas de las carpetas que contienen los audios de cada persona
DATASET_ROOT = r"C:\Users\alvar\Documents\Udemy\2_Curso_ Aprende Inteligencia Artificial y Deep Learning con Python &\7. Clasificación+de+audio+con+el+Perceptrón+Multicapa+I.ipynb\16000_pcm_speeches\16000_pcm_speeches"
BENJAMIN_DATA = os.path.join(DATASET_ROOT, "Benjamin_Netanyau")
JENS_DATA = os.path.join(DATASET_ROOT, "Jens_stoltenberg")
JULIA_DATA = os.path.join(DATASET_ROOT, "Julia_Gillard")
MARGARET_DATA = os.path.join(DATASET_ROOT, "Margaret_Tarcher")
NELSON_DATA = os.path.join(DATASET_ROOT, "Nelson_Mandela")

# Importamos las bibliotecas necesarias
# pydub: Para procesar y reproducir archivos de audio
# pydub.playback: Para la reproducción del audio

# Cargamos el audio usando pydub
# Utilizamos la función 'AudioSegment.from_wav' para cargar el archivo de audio "22.wav" en la variable 'audio'
# audio = AudioSegment.from_wav(os.path.join(BENJAMIN_DATA, "22.wav"))

# Reproducimos el audio
# Utilizamos la función 'play' para reproducir el audio almacenado en la variable 'audio'
# play(audio)

# REPRESENTACIÓN DEL SONIDO I
# Cargamos audio con librería librosa
wav, sr = librosa.load(os.path.join(BENJAMIN_DATA, "22.wav"))

print(wav)
print(sr)

# Calculamos la longitud del audio con la tasa de muestreo y el tamaño total de la señal
long_audio = len(wav)/sr
print(f"La longitud del audio en segundos es: {long_audio}")

# # Graficamos la onda de sonido
# plt.plot(wav)
# plt.show()
#
# # Hacemos zoom en el ciclo
# plt.plot(wav[1000:1200])
# plt.show()

# Tasa de muestreo original
wav, sr = librosa.load(os.path.join(BENJAMIN_DATA, "22.wav"), sr=None)
print(f"Tasa de muestreo: {sr} Hz")

# PREPARACIÓN DEL CONJUNTO DE DATOS
# Definimos una función para parsear los datos
def parsear_dataset(dataset_paths):
    X = []
    y = []
    for index, dataset in enumerate(dataset_paths):
        print(f"[+] Parsing {dataset} data ...")
        for fname in os.listdir(dataset):
            wav, sr = librosa.load(os.path.join(dataset, fname), sr=None)
            X.append(wav)
            y.append(index)
    return(X, y)

X, y = parsear_dataset([BENJAMIN_DATA, JENS_DATA])

print(f"La longitud del conjunto de datos es: {len(X)}")

# DIVISIÓN DEL CONJUNTO DE DATOS
# Split el dataset (entrenamiento y pruebas)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

print(len(X_train))
print(len(X_test))

# CONSTRUCCIÓN DEL MODELO
clf = MLPClassifier(activation="logistic", hidden_layer_sizes=(10,), solver="sgd")
clf.fit(X_train, y_train)

# Realizamos predicción sobre el conjunto de datos de prueba
y_pred = clf.predict(X_test)

# Evaluamos con el f1-score
print(f1_score(y_test, y_pred))

# Hacemos la matriz de confusion
print(confusion_matrix(y_test, y_pred))