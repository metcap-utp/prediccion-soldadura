import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path

# Definir directorio base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PREDICCION_USG_10S_DIR = BASE_DIR / "prediccion_etiqueta_usg" / "10s"

# Cargar los tres CSV con las etiquetas correspondientes
plate_data = pd.read_csv(PREDICCION_USG_10S_DIR / "rutas_etiquetas_conjunto.csv")
electrode_data = pd.read_csv(PREDICCION_USG_10S_DIR / "rutas_etiquetas_electrode.csv")
polarity_data = pd.read_csv(PREDICCION_USG_10S_DIR / "rutas_etiquetas_polarity.csv")

# Unir los tres DataFrames según las rutas de los audios
data = pd.merge(plate_data, electrode_data, on='Audio Path', how='inner')
data = pd.merge(data, polarity_data, on='Audio Path', how='inner')

# Codificar las etiquetas
plate_encoder = LabelEncoder()
electrode_encoder = LabelEncoder()
polarity_encoder = LabelEncoder()

data['Plate Thickness'] = plate_encoder.fit_transform(data['Plate Thickness'])
data['Electrode'] = electrode_encoder.fit_transform(data['Electrode'])
data['Polarity'] = polarity_encoder.fit_transform(data['Polarity'])

# Función para convertir audio a espectrograma log-mel compatible con VGGish
def load_vggish_logmel(file_path, num_frames=96):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    # Espectrograma log-mel con parámetros VGGish
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=400,
        hop_length=160,
        n_mels=64,
        fmin=125,
        fmax=7500
    )
    log_mel = librosa.power_to_db(mel_spec)

    # Ajustar forma a [num_frames, 64]
    if log_mel.shape[1] < num_frames:
        pad_width = num_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
    else:
        log_mel = log_mel[:, :num_frames]

    return log_mel.T  # Transponer para [num_frames, 64]

# Preprocesar los audios
data['Features'] = data['Audio Path'].apply(load_vggish_logmel)

# Convertir a arrays
X = np.stack(data['Features'])

# Las etiquetas (Placa, Electrodo, Polaridad) se usan como salida
y_plate = data['Plate Thickness'].values
y_electrode = data['Electrode'].values
y_polarity = data['Polarity'].values

# Separar en entrenamiento y prueba
X_train, X_test, y_plate_train, y_plate_test, y_electrode_train, y_electrode_test, y_polarity_train, y_polarity_test = train_test_split(
    X, y_plate, y_electrode, y_polarity, test_size=0.2, random_state=42)

# Crear modelo con entradas tipo VGGish: (96, 64)
input_layer = keras.layers.Input(shape=(96, 64))  # Entrada [frames, mels]

# Capas de convolución y pooling
x = keras.layers.Conv1D(64, kernel_size=3, activation='relu')(input_layer)
x = keras.layers.MaxPooling1D(pool_size=2)(x)
x = keras.layers.Conv1D(128, kernel_size=3, activation='relu')(x)
x = keras.layers.GlobalMaxPooling1D()(x)
x = keras.layers.Dense(128, activation='relu')(x)

# Tres salidas para cada etiqueta
output_plate = keras.layers.Dense(len(plate_encoder.classes_), activation='softmax', name='plate_output')(x)
output_electrode = keras.layers.Dense(len(electrode_encoder.classes_), activation='softmax', name='electrode_output')(x)
output_polarity = keras.layers.Dense(len(polarity_encoder.classes_), activation='softmax', name='polarity_output')(x)

# Modelo
model = keras.Model(inputs=input_layer, outputs=[output_plate, output_electrode, output_polarity])

# Compilar modelo
model.compile(optimizer='adam',
              loss={'plate_output': 'sparse_categorical_crossentropy', 
                    'electrode_output': 'sparse_categorical_crossentropy',
                    'polarity_output': 'sparse_categorical_crossentropy'},
              metrics={'plate_output': 'accuracy', 
                       'electrode_output': 'accuracy', 
                       'polarity_output': 'accuracy'})

# Entrenar modelo
model.fit(X_train, 
          {'plate_output': y_plate_train, 'electrode_output': y_electrode_train, 'polarity_output': y_polarity_train},
          epochs=100, 
          batch_size=32,
          validation_data=(X_test, 
                           {'plate_output': y_plate_test, 'electrode_output': y_electrode_test, 'polarity_output': y_polarity_test}))

# Evaluar modelo
evaluation_results = model.evaluate(X_test, 
                                   {'plate_output': y_plate_test, 
                                    'electrode_output': y_electrode_test, 
                                    'polarity_output': y_polarity_test})

test_loss, test_loss_plate, test_loss_electrode, test_loss_polarity, test_accuracy_plate, test_accuracy_electrode, test_accuracy_polarity = evaluation_results

print(f'Test accuracy Plate: {test_accuracy_plate:.2f}%')
print(f'Test accuracy Electrode: {test_accuracy_electrode:.2f}%')
print(f'Test accuracy Polarity: {test_accuracy_polarity:.2f}%')

# Guardar modelo en formato nativo Keras
model.save(PREDICCION_USG_10S_DIR / "my_model_vggish_completo.keras")
