import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

# DATASETS
DATASET_ROOT = r"C:\Users\alvar\Documents\Udemy\2_Curso_ Aprende Inteligencia Artificial y Deep Learning con Python &\7. Clasificación+de+audio+con+el+Perceptrón+Multicapa+I.ipynb\16000_pcm_speeches\16000_pcm_speeches"
BENJAMIN_DATA = os.path.join(DATASET_ROOT, "Benjamin_Netanyau")
JENS_DATA = os.path.join(DATASET_ROOT, "Jens_Stoltenberg")
JULIA_DATA = os.path.join(DATASET_ROOT, "Julia_Gillard")
MARGARET_DATA = os.path.join(DATASET_ROOT, "Magaret_Tarcher")
NELSON_DATA = os.path.join(DATASET_ROOT, "Nelson_Mandela")

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

# REPRESENTACIÓN DEL SONIDO II
signal = np.cos(np.arange(0, 20, 0.2))

# plt.plot(signal)
# plt.show()

signal = 2*np.cos(np.arange(0, 20, 0.2)*2)

# plt.plot(signal)
# plt.show()

cos1 = np.cos(np.arange(0, 20, 0.2))
cos2 = 2*np.cos(np.arange(0, 20, 0.2)*2)
cos3 = 8*np.cos(np.arange(0, 20, 0.2)*4)

signal = cos1 + cos2 + cos3

# plt.plot(signal)
# plt.show()

fft = np.fft.fft(signal)[:50]
fft = np.abs(fft)

# plt.plot(fft)
# plt.show()

D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)

librosa.display.specshow(D, y_axis="linear")
# plt.show()

# print(D.shape)

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

X_prep, y_prep = parsear_dataset([JENS_DATA, JULIA_DATA, MARGARET_DATA, NELSON_DATA])

# DIVISIÓN DEL CONJUNTO DE DATOS
# Split el dataset (entrenamiento y pruebas)
X_train, X_test, y_train, y_test = train_test_split(X_prep, y_prep, test_size=0.05)

print(len(X_train))
print(len(X_test))

# CONSTRUCCIÓN DEL MODELO
# Preprocesamos los subconjuntos de datos para nuestra red neuronal
X_train_prep = np.array(X_train).astype("float32") / 255
y_train_prep = np.array(y_train)

X_test_prep = np.array(X_test).astype("float32") / 255
y_test_prep = np.array(y_test)

# Entrenamos nuestro perceptron
clf = MLPClassifier(activation="logistic", hidden_layer_sizes=(10,), solver="sgd")
clf.fit(X_train, y_train)

# Realizamos predicción sobre el conjunto de datos de prueba
y_pred = clf.predict(X_test)

# Evaluamos con el f1-score
print(f1_score(y_test, y_pred, average="weighted"))