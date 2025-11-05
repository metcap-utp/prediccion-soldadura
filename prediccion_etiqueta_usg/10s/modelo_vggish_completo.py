import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import pandas as pd
import librosa
import os
import random
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Definir directorio base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PREDICCION_USG_10S_DIR = BASE_DIR / "prediccion_etiqueta_usg" / "10s"
VGGISH_MODEL_DIR = BASE_DIR / "prediccion_etiqueta_cc" / "vggish_1"

# Cargar el modelo VGGish desde TensorFlow Hub
vggish_model = hub.load(str(VGGISH_MODEL_DIR))

# Cargar el modelo personalizado previamente entrenado
your_model = tf.keras.models.load_model(
    PREDICCION_USG_10S_DIR / "my_model_vggish_completo.keras"
)

# Cargar los datos CSV (Tres CSV para las tres etiquetas)
plate_data = pd.read_csv(PREDICCION_USG_10S_DIR / "rutas_etiquetas_conjunto.csv")
electrode_data = pd.read_csv(PREDICCION_USG_10S_DIR / "rutas_etiquetas_electrode.csv")
polarity_data = pd.read_csv(PREDICCION_USG_10S_DIR / "rutas_etiquetas_polarity.csv")

# Unir los tres DataFrames según las rutas de los audios
data = pd.merge(plate_data, electrode_data, on='Audio Path', how='inner')
data = pd.merge(data, polarity_data, on='Audio Path', how='inner')

# Obtener las rutas de archivo de audio (columna 'Audio') y etiquetas (columnas 'plate', 'electrode', 'polarity')
X = data['Audio Path']
y_plate = data['Plate Thickness']
y_electrode = data['Electrode']
y_polarity = data['Polarity']

# Convertir etiquetas de texto a valores numéricos
plate_encoder = LabelEncoder()
electrode_encoder = LabelEncoder()
polarity_encoder = LabelEncoder()

y_plate = plate_encoder.fit_transform(y_plate)
y_electrode = electrode_encoder.fit_transform(y_electrode)
y_polarity = polarity_encoder.fit_transform(y_polarity)

# Almacenar las clases en una variable
plate_classes = plate_encoder.classes_
electrode_classes = electrode_encoder.classes_
polarity_classes = polarity_encoder.classes_

# Función para predecir usando solo VGGish
def predict_band_name(audio_path):
    # Cargar y preprocesar el audio
    audio, sr = librosa.load(audio_path, sr=8000)
    
    # Obtener las características de VGGish
    vggish_features = vggish_model(audio)
    
    # Ajustar la forma de las características para que coincidan con la entrada esperada por el modelo
    if vggish_features.shape[1] != 96 or vggish_features.shape[2] != 64:
        # Ajuste de la forma utilizando np.resize o un padding si es necesario
        vggish_features_resized = np.resize(vggish_features, (1, 96, 64))  # Ajustar las dimensiones
    else:
        vggish_features_resized = np.expand_dims(vggish_features, axis=0)
    
    # Realizar la predicción usando las características de VGGish
    prediction = your_model.predict(vggish_features_resized)

    # Obtener las etiquetas predichas
    plate_pred = plate_classes[np.argmax(prediction[0])]
    electrode_pred = electrode_classes[np.argmax(prediction[1])]
    polarity_pred = polarity_classes[np.argmax(prediction[2])]

    return plate_pred, electrode_pred, polarity_pred

# Obtener archivos de audio y sus etiquetas reales
def get_file_list_and_labels(directory):
    audio_files = []
    labels = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav'):
                path = os.path.join(root, file)
                audio_files.append(path)
                partes = path.split(os.sep)
                if len(partes) >= 7:  # Asegúrate de que la ruta tenga al menos 7 partes
                    plate = partes[-5]  # .../Plate_3mm/...
                    electrode = partes[-4]  # .../E6010/...
                    polarity = partes[-3]  # .../DC/...
                    labels.append((plate, electrode, polarity))
    return audio_files, labels

# Directorios de audio
AUDIOS_DIR = BASE_DIR / "prediccion_etiqueta_cc" / "audios"
directory_train = str(AUDIOS_DIR / "train")
directory_test = str(AUDIOS_DIR / "test")

train_files, train_labels = get_file_list_and_labels(directory_train)
test_files, test_labels = get_file_list_and_labels(directory_test)

# ---------------------------------------------
# Seleccionar muestra aleatoria de prueba
test_files_sampled = random.sample(test_files, 10)
test_labels_sampled = []

# Crear lista para guardar los resultados
results = []

# ---------------------------------------------
# Realizar predicciones sobre la muestra
correctas = 0
correctas_plate = 0
correctas_electrode = 0
correctas_polarity = 0

print("Predicciones y etiquetas reales:\n")

for archivo in test_files_sampled:
    partes = archivo.split(os.sep)
    
    # Asegurarse de que se extraen correctamente las etiquetas
    if len(partes) >= 7:
        etiqueta_real_plate = partes[-5]  # .../Plate_3mm/...
        etiqueta_real_electrode = partes[-4]  # .../E6010/...
        etiqueta_real_polarity = partes[-3]  # .../DC/...
    else:
        continue  # Si la ruta no tiene el formato esperado, omitimos el archivo
    
    test_labels_sampled.append((etiqueta_real_plate, etiqueta_real_electrode, etiqueta_real_polarity))

    prediccion_plate, prediccion_electrode, prediccion_polarity = predict_band_name(archivo)

    results.append([archivo, etiqueta_real_plate, etiqueta_real_electrode, etiqueta_real_polarity,
                   prediccion_plate, prediccion_electrode, prediccion_polarity])

    if (prediccion_plate == etiqueta_real_plate):
        correctas_plate += 1
    if (prediccion_electrode == etiqueta_real_electrode):
        correctas_electrode += 1
    if (prediccion_polarity == etiqueta_real_polarity):
        correctas_polarity += 1

    if (prediccion_plate == etiqueta_real_plate and
        prediccion_electrode == etiqueta_real_electrode and
        prediccion_polarity == etiqueta_real_polarity):
        correctas += 1

    print(f"Archivo: {archivo} | Real: {etiqueta_real_plate}, {etiqueta_real_electrode}, {etiqueta_real_polarity} | Predicción: {prediccion_plate}, {prediccion_electrode}, {prediccion_polarity}")

# Mostrar resultados de precisión
accuracy_total = correctas / len(test_files_sampled)
accuracy_plate = correctas_plate / len(test_files_sampled)
accuracy_electrode = correctas_electrode / len(test_files_sampled)
accuracy_polarity = correctas_polarity / len(test_files_sampled)

print(f"\nAccuracy total sobre muestra de prueba (todas las etiquetas correctas): {accuracy_total:.2f}")
print(f"Accuracy Plate: {accuracy_plate:.2f}")
print(f"Accuracy Electrode: {accuracy_electrode:.2f}")
print(f"Accuracy Polarity: {accuracy_polarity:.2f}")
