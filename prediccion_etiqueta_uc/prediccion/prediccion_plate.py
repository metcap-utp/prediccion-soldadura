import os
import pandas as pd
import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import tensorflow_hub as hub


# Función para obtener archivos 'drums_louder.wav' dentro de subcarpetas
def get_file_list(directory):
    file_list = []
    labels = []  # Almacenar la etiqueta de cada archivo
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if (
                filename == "drums_louder.wav"
            ):  # Solo buscar 'drums_louder.wav'
                file_path = os.path.join(root, filename)
                file_list.append(file_path)

                # Extraer la etiqueta desde la carpeta (etiqueta principal)
                parent_folder = os.path.basename(os.path.dirname(root))
                if parent_folder in ["Plate_3mm", "Plate_6mm", "Plate_12mm"]:
                    labels.append(parent_folder)

    return file_list, labels


# Cargar el modelo VGGish desde el directorio de guardado (asumiendo CWD = prediccion_etiqueta_uc)
vggish_model = tf.saved_model.load("prediccion/vggish_1")

# Cargar el modelo previamente entrenado desde un archivo .h5 (asumiendo CWD = prediccion_etiqueta_uc)
model = tf.keras.models.load_model("my_model_completo01.h5")

#  Definir las clases
classes = ["Plate_3mm", "Plate_6mm", "Plate_12mm"]

#  Cargar archivos de entrenamiento
train_files, train_labels = get_file_list("audios02/train")
if not train_files:
    raise ValueError(
        "No se encontraron archivos en la carpeta de entrenamiento."
    )

#  Crear DataFrame de entrenamiento
train = pd.DataFrame({"filename": train_files, "label": train_labels})

#  Cargar archivos de prueba
test_files, test_labels = get_file_list("audios02/test")
if not test_files:
    raise ValueError("No se encontraron archivos en la carpeta de prueba.")

#  Crear DataFrame de prueba
test = pd.DataFrame({"filename": test_files, "label": test_labels})


#  Función para extraer características MFCC de los archivos de audio
def load_and_preprocess_audio(file_path, desired_length=5000):
    # Cargar audio con Librosa
    y, sr = librosa.load(file_path, sr=8000)

    # Aplicar preprocesamiento: MFCC (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Normalizar las características
    mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

    # Ajustar la longitud deseada
    if mfccs.shape[1] < desired_length:
        # Si la longitud de mfccs es menor que la deseada, se rellena con ceros
        mfccs_padded = np.pad(
            mfccs,
            ((0, 0), (0, desired_length - mfccs.shape[1])),
            mode="constant",
        )
    else:
        # Si la longitud de mfccs es mayor, se corta para que tenga la longitud deseada
        mfccs_padded = mfccs[:, :desired_length]

    # Ajustar la forma a (13, desired_length)
    return mfccs_padded


#  Función para predecir el género de un audio utilizando el modelo VGGish y luego el modelo entrenado (.h5)
def predict_genre(filenames):
    results = []
    for file in filenames:
        # Cargar el archivo de audio usando librosa
        waveform, sr = librosa.load(file, sr=16000)

        # Si la longitud no es un múltiplo exacto de 16000, añadir ceros al final
        if waveform.shape[0] % 16000 != 0:
            waveform = np.concatenate([waveform, np.zeros(16000)])

        # Preprocesar el audio: VGGish espera que la entrada sea un tensor de 1D
        inp = tf.constant(waveform, dtype="float32")

        # Extraer características de audio usando VGGish
        vggish_features = vggish_model(inp)[0].numpy()

        # Redimensionar las características de VGGish para que coincidan con la entrada del modelo
        vggish_features = vggish_features[
            :58
        ]  # Tomar los primeros 58 valores de las características
        vggish_features = np.expand_dims(
            vggish_features, axis=-1
        )  # Redimensionar a (58, 1)

        # Realizar la predicción usando el modelo entrenado (.h5)
        prediction = model.predict(
            np.expand_dims(vggish_features, axis=0)
        )  # Ajustar si el modelo requiere otro formato
        predicted_class = classes[np.argmax(prediction)]

        results.append(
            predicted_class
        )  # Devolver la clase con el puntaje más alto

    return results


#  Hacer predicciones en el conjunto de prueba
test["prediction"] = predict_genre(test["filename"])

#  Hacer predicciones en el conjunto de entrenamiento
train["prediction"] = predict_genre(train["filename"])

#  Evaluación del modelo en el conjunto de prueba y entrenamiento
print("Reporte de clasificación en entrenamiento:")
print(classification_report(train["label"], train["prediction"]))

print("Reporte de clasificación en prueba:")
print(classification_report(test["label"], test["prediction"]))
