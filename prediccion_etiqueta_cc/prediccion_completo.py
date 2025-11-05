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


# Función para extraer características VGGish (128 dimensiones)
def extract_features_with_vggish(audio_path):
    """
    Extrae embeddings VGGish de un archivo de audio.

    Args:
        audio_path: Ruta al archivo de audio

    Returns:
        Array de 128 características VGGish
    """
    try:
        # Cargar audio a 16kHz (requerido por VGGish)
        y_audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # VGGish espera entrada 1D directa (sin reshape)
        vggish_features = vggish_model(y_audio)

        # Promediar sobre el eje temporal
        vggish_avg = np.mean(vggish_features.numpy(), axis=0)

        return vggish_avg
    except Exception as e:
        print(f"Error extrayendo VGGish de {audio_path}: {e}")
        return np.zeros(128)


# Función para extraer características MFCC (40 dimensiones: 20 mean + 20 std)
def extract_mfcc(audio_path, n_mfcc=20):
    """
    Extrae coeficientes MFCC de un archivo de audio.

    Args:
        audio_path: Ruta al archivo de audio
        n_mfcc: Número de coeficientes MFCC a extraer

    Returns:
        Array de 40 características (20 medias + 20 desviaciones estándar)
    """
    try:
        # Cargar audio a 16kHz
        y_audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Extraer MFCCs
        mfccs = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc)

        # Calcular media y desviación estándar
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Concatenar
        mfcc_features = np.concatenate([mfcc_mean, mfcc_std])

        return mfcc_features
    except Exception as e:
        print(f"Error extrayendo MFCC de {audio_path}: {e}")
        return np.zeros(40)


# Función para combinar todas las características (168 dimensiones totales)
def extract_combined_features(audio_path):
    """
    Extrae características combinadas VGGish + MFCC.
    Debe coincidir con el proceso de extracción del entrenamiento.

    Args:
        audio_path: Ruta al archivo de audio

    Returns:
        Array de 168 características (128 VGGish + 40 MFCC)
    """
    # Extraer VGGish (128)
    vggish_features = extract_features_with_vggish(audio_path)

    # Extraer MFCC (40)
    mfcc_features = extract_mfcc(audio_path)

    # Combinar: 128 + 40 = 168
    combined_features = np.concatenate([vggish_features, mfcc_features])

    # Redimensionar para el modelo Conv1D
    return combined_features.reshape((168, 1))


# Función para predecir todas las etiquetas
def predict_all_labels(audio_path):
    """
    Predice todas las etiquetas (Plate Thickness, Electrode, Polarity)
    para un archivo de audio dado.

    Args:
        audio_path: Ruta al archivo de audio

    Returns:
        Diccionario con las etiquetas predichas
    """
    # Extraer características (168 dimensiones)
    features = extract_combined_features(audio_path)

    # Verificar dimensión correcta
    if features.shape[0] != 168:
        print(
            f"Advertencia: Características={features.shape[0]}, esperadas=168"
        )
        return {}

    # Predecir con el modelo
    predictions = your_model.predict(
        np.expand_dims(features, axis=0), verbose=0
    )

    # El modelo devuelve un array con 3 valores (uno por cada etiqueta)
    # predictions shape: (1, 3) -> [plate_thickness_prob, electrode_prob, polarity_prob]
    results = {}

    # Decodificar cada predicción
    for idx, label in enumerate(y_labels):
        # Para multi-label con sigmoid, el valor > 0.5 se considera positivo
        # Pero en este caso, usamos argmax sobre las clases
        label_prediction = predictions[0][idx]
        # Como usamos LabelBinarizer, el valor es 0 o 1
        class_idx = int(round(label_prediction))
        results[label] = (
            label_classes[label][class_idx]
            if class_idx < len(label_classes[label])
            else label_classes[label][0]
        )

    return results


# Función para extraer etiquetas reales de la ruta
def extract_labels_from_path(path):
    parts = path.split("/")
    for i, part in enumerate(parts):
        if "Placa_" in part:
            return {
                "Plate Thickness": part,
                "Electrode": parts[i + 1] if i + 1 < len(parts) else "Unknown",
                "Polarity": parts[i + 2] if i + 2 < len(parts) else "Unknown",
            }
    return {}


# Lista de archivos de prueba
test_files = [
    PREDICCION_CC_DIR
    / "audios"
    / "train"
    / "Placa_12mm"
    / "E7018"
    / "DC"
    / "240926-144305_Audio"
    / "drums_louder.wav",
    PREDICCION_CC_DIR
    / "audios"
    / "train"
    / "Placa_6mm"
    / "E6013"
    / "AC"
    / "240923-130248_Audio"
    / "drums_louder.wav",
    PREDICCION_CC_DIR
    / "audios"
    / "train"
    / "Placa_3mm"
    / "E6010"
    / "DC"
    / "240905-155358_Audio"
    / "drums_louder.wav",
]

print("=" * 70)
print("PREDICCIÓN DE ETIQUETAS - MODELO COMPLETO")
print("=" * 70)

for audio_file_path in test_files:
    audio_file_path = str(audio_file_path)

    # Verificar si el archivo existe
    if not Path(audio_file_path).exists():
        print(f"\n[!] Archivo no encontrado: {audio_file_path}")
        continue

    try:
        print(f"\n{'─' * 70}")
        print(f"Archivo: {audio_file_path.split('/')[-2]}")

        # Extraer etiquetas reales de la ruta
        real_labels = extract_labels_from_path(audio_file_path)

        # Predecir etiquetas
        predictions = predict_all_labels(audio_file_path)

        # Mostrar comparación
        print(
            f"\n{'Etiqueta':<20} {'Real':<15} {'Predicho':<15} {'Correcta':<10}"
        )
        print("─" * 70)

        for label in y_labels:
            real = real_labels.get(label, "N/A")
            pred = predictions.get(label, "N/A")
            match = "OK" if real == pred else "FAIL"
            print(f"{label:<20} {real:<15} {pred:<15} {match:<10}")

    except Exception as e:
        print(f"\n[ERROR] Error en prediccion: {e}")
        import traceback

        traceback.print_exc()

print(f"\n{'═' * 70}")
