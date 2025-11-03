import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# Definir directorio base
BASE_DIR = Path(__file__).resolve().parent.parent
PREDICCION_CC_DIR = BASE_DIR / "prediccion_etiqueta_cc"

# Cargar el modelo VGGish
vggish_model_handle = str(PREDICCION_CC_DIR / "vggish_1")
vggish_model = hub.load(vggish_model_handle)

# Cargar el modelo entrenado (formato nativo Keras)
your_model = tf.keras.models.load_model(
    PREDICCION_CC_DIR / "my_model_completo.keras"
)

# Cargar el CSV con rutas de audio y etiquetas
data = pd.read_csv(PREDICCION_CC_DIR / "rutas_etiquetas_completos.csv")

# Extraer rutas de audio y etiquetas
X = data["Audio Path"]
y_labels = ["Plate Thickness", "Electrode", "Polarity"]
label_encoders = {}
label_classes = {}

# Convertir etiquetas categóricas a numéricas
for label in y_labels:
    encoder = LabelEncoder()
    data[label] = encoder.fit_transform(data[label])
    label_encoders[label] = encoder
    label_classes[label] = encoder.classes_


# Función para ajustar características


def pad_or_truncate(features, desired_length=10):
    features = features.flatten()
    current_length = len(features)
    if current_length > desired_length:
        features = features[:desired_length]
    elif current_length < desired_length:
        padding = np.zeros(desired_length - current_length)
        features = np.concatenate((features, padding))
    return features.reshape((desired_length, 1))


# Función para extraer características con VGGish
def extract_features_with_vggish(audio_path):
    try:
        y_audio, sr = librosa.load(audio_path, sr=16000)
        y_audio = y_audio.flatten()
        if len(y_audio) % 1024 != 0:
            padding = 1024 - (len(y_audio) % 1024)
            y_audio = np.pad(y_audio, (0, padding), mode="constant")
        audio_input = tf.convert_to_tensor(y_audio, dtype=tf.float32)
        vggish_features = vggish_model(audio_input)
        return vggish_features.numpy()
    except Exception as e:
        print(f"Error con {audio_path}: {e}")
        return np.zeros((128,))


# Función para extraer características MFCC
def extract_mfcc(audio_path, n_mfcc=10):
    y_audio, sr = librosa.load(audio_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs.flatten()[:10]


# Función para combinar todas las características
def extract_combined_features(audio_path):
    vggish_features = extract_features_with_vggish(audio_path)
    mfcc_features = extract_mfcc(audio_path)
    combined_features = np.concatenate(
        [vggish_features.flatten(), mfcc_features]
    )
    return pad_or_truncate(combined_features, desired_length=10)


# Función para predecir etiquetas
def predict_label(audio_path, encoder):
    features = extract_combined_features(audio_path)
    prediction = your_model.predict(np.expand_dims(features, axis=0))
    return encoder[np.argmax(prediction)]


# Función para extraer características adicionales (ZCR, centroides espectrales)
def extract_additional_features(audio_path):
    y_audio, sr = librosa.load(audio_path, sr=16000)
    zcr = librosa.feature.zero_crossing_rate(y_audio)
    spectral_centroid = librosa.feature.spectral_centroid(y=y_audio, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y_audio, sr=sr)
    return np.concatenate(
        [zcr, spectral_centroid, spectral_rolloff], axis=0
    ).flatten()


# Prueba con un archivo de audio específico
audio_file_path = str(
    PREDICCION_CC_DIR
    / "audios"
    / "train"
    / "Placa_12mm"
    / "E7018"
    / "DC"
    / "240926-144305_Audio"
    / "drums_louder.wav"
)
# audio_file_path = str(PREDICCION_CC_DIR / 'audios' / 'train' / 'Placa_6mm' / 'E6011DC' / 'DC' / '240905-140049_Audio' / 'drums_louder.wav')
# audio_file_path = str(PREDICCION_CC_DIR / 'audios' / 'train' / 'Placa_6mm' / 'E7018DC' / 'DC' / '240826-131533_Audio' / 'drums_louder.wav')
# audio_file_path = str(PREDICCION_CC_DIR / 'audios' / 'train' / 'Placa_3mm' / 'E6013AC' / 'AC' / '240812-144347_Audio' / 'drums_louder.wav')

try:
    for label in y_labels:
        prediction = predict_label(audio_file_path, label_classes[label])
        print(f"{label}: {prediction}")
except Exception as e:
    print(f"Error en predicción: {e}")
