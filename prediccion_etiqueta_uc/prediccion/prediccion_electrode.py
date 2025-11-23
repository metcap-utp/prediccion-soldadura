import os

import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Directorio base del módulo UC
script_dir = os.path.dirname(os.path.abspath(__file__))
uc_dir = os.path.join(script_dir, "..")


# Evitar OOM en GPU: permitir memory growth en las GPUs disponibles
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            # Si falla, no interrumpir — la configuración puede no ser soportada
            pass


# Nota: no definir `classes` de forma estática aquí — se detectarán
# automáticamente a partir de las carpetas en audios01/train y audios01/test


# Función para obtener archivos 'drums_louder.wav' dentro de subcarpetas
def get_file_list(directory):
    file_list = []
    labels = []  # Almacenar la etiqueta de cada archivo
    detected_labels = set()

    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename == "drums_louder.wav":
                # Obtener la parte relativa desde el directorio base
                rel = os.path.relpath(root, directory)
                parts = rel.split(os.sep)

                # Preferir la primera parte de la ruta relativa como etiqueta (más robusto)
                if len(parts) >= 1 and parts[0] not in (".", ""):
                    label = parts[0]
                else:
                    # Fallback: carpeta padre del directorio que contiene el archivo
                    label = os.path.basename(os.path.dirname(root))

                detected_labels.add(label)

                file_path = os.path.join(root, filename)
                file_list.append(file_path)
                labels.append(label)

    # Si no se encontraron archivos, devolver listas vacías
    if not file_list:
        return file_list, labels

    # Comprobar si las etiquetas encontradas coinciden con las clases esperadas
    allowed = set(classes) if classes else set()
    if allowed and not (detected_labels & allowed):
        # Ninguna etiqueta detectada coincide con `classes` — avisar y usar las detectadas
        print(
            f"Warning: ninguna etiqueta detectada en '{directory}' coincide con la lista `classes`."
        )
        print(f"Etiquetas detectadas: {sorted(detected_labels)}")
        print("Usando las etiquetas detectadas como etiquetas finales (fallback).")
        # No filtramos: mantenemos file_list/labels tal cual
        return file_list, labels

    # Si hay mezcla, filtrar solo los archivos cuyas etiquetas están en `classes`
    if allowed:
        filtered_files = []
        filtered_labels = []
        for f, l in zip(file_list, labels):
            if l in allowed:
                filtered_files.append(f)
                filtered_labels.append(l)
        return filtered_files, filtered_labels

    return file_list, labels


print("=" * 60)
print("PREDICCION - ELECTRODE")
print("=" * 60)

# Cargar el modelo VGGish desde el directorio de guardado
print("\n>> Cargando extractor de características VGGish...")
vggish_path = os.path.join(uc_dir, "prediccion", "vggish_1")
vggish_model = tf.saved_model.load(vggish_path)
print("   [OK] VGGish cargado")

# Cargar el modelo previamente entrenado (formato nativo Keras)
print(">> Cargando clasificador entrenado: my_model_completo01.keras")
model_path = os.path.join(uc_dir, "my_model_completo01.keras")
model = tf.keras.models.load_model(model_path)
# Determinar la dimensión de salida del modelo
try:
    output_dim = model.output_shape[-1]
    print(f"   [OK] Modelo cargado - Dimensión de salida: {output_dim}")
except Exception:
    output_dim = None


# Definir las clases estáticas (los scripts esperan estas etiquetas)
classes = [
    "E6010",
    "E6011",
    "E6013",
    "E7018",
]  # Cambiar si tienes más etiquetas


# Cargar rutas y etiquetas desde CSV (usar el CSV maestro en la raíz del proyecto)
# El CSV debe contener una columna con la ruta: 'Audio Path' y una columna de etiqueta.
print("\n>> Leyendo archivo de etiquetas: rutas_etiquetas_electrode.csv")
csv_path = os.path.join(uc_dir, "rutas_etiquetas_electrode.csv")
df_all = pd.read_csv(csv_path)
print(f"   [OK] {len(df_all)} archivos de audio encontrados en el CSV")

# Determinar la columna de etiqueta disponible (prioridad)
if "Electrode" in df_all.columns:
    label_col = "Electrode"
elif "Type of Current" in df_all.columns:
    label_col = "Type of Current"
elif "label" in df_all.columns:
    label_col = "label"
else:
    raise ValueError(
        "No se encontró columna de etiquetas en rutas_etiquetas_electrode.csv (se buscó Electrode/Type of Current/label)."
    )

# Filtrar filas para train/test por la ruta (las rutas en el CSV contienen 'audios01/train' o 'audios01/test')
mask_train = df_all["Audio Path"].astype(str).str.contains("audios01/train")
mask_test = df_all["Audio Path"].astype(str).str.contains("audios01/test")

df_train = df_all[mask_train].copy()
df_test = df_all[mask_test].copy()

if df_train.empty:
    raise ValueError(
        "No se encontraron filas de entrenamiento en rutas_etiquetas_electrode.csv para 'audios01/train'."
    )
if df_test.empty:
    # Intento 1: mapear archivos existentes en disco dentro de audios01/test a etiquetas del CSV
    print("\n[AVISO] No se encontraron datos de prueba en el CSV")
    print(">> Buscando archivos de audio en el directorio audios01/test/...")
    test_dir = os.path.join(uc_dir, "audios01", "test")
    test_files_fs, _ = get_file_list(test_dir)
    mapped_files = []
    mapped_labels = []

    def _norm(p):
        return os.path.normpath(str(p)).lstrip("./")

    df_all["_path_norm"] = df_all["Audio Path"].astype(str).apply(_norm)

    for f in test_files_fs:
        fn = _norm(f)
        matches = df_all[df_all["_path_norm"].str.endswith(fn)]
        if not matches.empty:
            mapped_files.append(f)
            mapped_labels.append(matches.iloc[0][label_col])
        else:
            base = os.path.basename(f)
            matches2 = df_all[df_all["Audio Path"].astype(str).str.endswith(base)]
            if not matches2.empty:
                mapped_files.append(f)
                mapped_labels.append(matches2.iloc[0][label_col])

    if mapped_files:
        print(
            f"   [OK] {len(mapped_files)} archivos mapeados exitosamente con etiquetas del CSV"
        )
        test = pd.DataFrame({"filename": mapped_files, "label": mapped_labels})
        df_test_fallback_used = True
    else:
        # Intento 2: crear un split desde las filas de Train en CSV
        print(">> No se pudieron mapear archivos. Dividiendo datos de entrenamiento...")
        print("   Creando split aleatorio: 80% entrenamiento, 20% prueba")
        df_train = df_train.sample(frac=1.0, random_state=42)
        small_test = df_train.sample(frac=0.2, random_state=42)
        df_train = df_train.drop(small_test.index)
        train = pd.DataFrame(
            {
                "filename": df_train["Audio Path"].tolist(),
                "label": df_train[label_col].tolist(),
            }
        )
        test = pd.DataFrame(
            {
                "filename": small_test["Audio Path"].tolist(),
                "label": small_test[label_col].tolist(),
            }
        )
        df_test_fallback_used = True
        print(f"   [OK] Conjuntos creados - Train: {len(train)} | Test: {len(test)}")
else:
    df_test_fallback_used = False

# Crear DataFrames de trabajo usando las rutas y etiquetas desde el CSV
# (solo si el fallback no fue usado)
if not df_test_fallback_used:
    train = pd.DataFrame(
        {
            "filename": df_train["Audio Path"].tolist(),
            "label": df_train[label_col].tolist(),
        }
    )
    test = pd.DataFrame(
        {
            "filename": df_test["Audio Path"].tolist(),
            "label": df_test[label_col].tolist(),
        }
    )
else:
    # Asegurar que train también se cree desde df_train si viene del CSV
    if "train" not in locals():
        train = pd.DataFrame(
            {
                "filename": df_train["Audio Path"].tolist(),
                "label": df_train[label_col].tolist(),
            }
        )

print(f"\n>> Configuración de clases para predicción: {classes}")
# Determinar si el modelo es binario (salida escalar)
is_binary_model = output_dim == 1
modelo_tipo = "binario" if is_binary_model else "multiclase"
print(f"   Tipo de modelo detectado: {modelo_tipo}")
if is_binary_model and len(classes) != 2:
    print("   [AVISO] El modelo es binario pero hay más de 2 clases definidas")


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
        prediction = model.predict(np.expand_dims(vggish_features, axis=0))

        # Si el modelo es binario (salida escalar), usar umbral 0.5
        if is_binary_model:
            score = float(np.squeeze(prediction))
            # si hay dos clases, mapear 0->classes[0], 1->classes[1]
            predicted_class = classes[1] if score >= 0.5 else classes[0]
        else:
            # Multiclase: aplicar argmax
            predicted_class = classes[int(np.argmax(prediction))]

        results.append(predicted_class)  # Devolver la clase con el puntaje más alto

    return results


#  Hacer predicciones en el conjunto de prueba
print("\n>> Ejecutando predicciones en conjunto de PRUEBA...")
print(f"   Procesando {len(test)} archivos de audio...")
test["prediction"] = predict_genre(test["filename"])
print("   [OK] Predicciones completadas")

#  Hacer predicciones en el conjunto de entrenamiento
print("\n>> Ejecutando predicciones en conjunto de ENTRENAMIENTO...")
print(f"   Procesando {len(train)} archivos de audio...")
train["prediction"] = predict_genre(train["filename"])
print("   [OK] Predicciones completadas")

#  Evaluación del modelo en el conjunto de prueba y entrenamiento
print("\n" + "=" * 60)
print("REPORTE DE CLASIFICACION - CONJUNTO DE ENTRENAMIENTO")
print("=" * 60)
print(classification_report(train["label"], train["prediction"]))

print("\n" + "=" * 60)
print("REPORTE DE CLASIFICACION - CONJUNTO DE PRUEBA")
print("=" * 60)
print(classification_report(test["label"], test["prediction"]))
