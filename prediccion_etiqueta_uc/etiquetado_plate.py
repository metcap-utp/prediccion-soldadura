import os

import librosa
import numpy as np
import pandas as pd


# Función para extraer características acústicas
def extraer_caracteristicas(audio_path):
    y, sr = librosa.load(audio_path, sr=8000)

    length = len(y) / sr
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    harmony = librosa.effects.harmonic(y)
    perceptr = librosa.effects.percussive(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    features = [
        length,
        np.mean(chroma_stft),
        np.var(chroma_stft),
        np.mean(rms),
        np.var(rms),
        np.mean(spectral_centroid),
        np.var(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.var(spectral_bandwidth),
        np.mean(rolloff),
        np.var(rolloff),
        np.mean(zero_crossing_rate),
        np.var(zero_crossing_rate),
        np.mean(harmony),
        np.var(harmony),
        np.mean(perceptr),
        np.var(perceptr),
        tempo,
    ]

    for i in range(20):
        features.append(np.mean(mfccs[i]))
        features.append(np.var(mfccs[i]))

    return features


# Función para recorrer directorios y capturar rutas, etiquetas y características
def obtener_rutas_y_etiquetas_con_caracteristicas(directorio_base):
    datos = []

    for root, _, files in os.walk(directorio_base):
        for file in files:
            if (
                file == "drums_louder.wav"
            ):  # Filtra solo los archivos 'drums_louder.wav'
                audio_path = os.path.join(root, file)
                partes_ruta = os.path.relpath(audio_path, directorio_base).split(os.sep)

                try:
                    # Extraer la etiqueta de la carpeta principal
                    plate_thickness = partes_ruta[0]

                    # Extraer características acústicas
                    características = extraer_caracteristicas(audio_path)
                    datos.append([audio_path, plate_thickness] + características)
                except IndexError:
                    continue

    # Definir las columnas
    columnas = [
        "Audio Path",
        "Plate Thickness",
        "length",
        "chroma_stft_mean",
        "chroma_stft_var",
        "rms_mean",
        "rms_var",
        "spectral_centroid_mean",
        "spectral_centroid_var",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_var",
        "rolloff_mean",
        "rolloff_var",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_var",
        "harmony_mean",
        "harmony_var",
        "perceptr_mean",
        "perceptr_var",
        "tempo",
    ]

    for i in range(1, 21):
        columnas.append(f"mfcc{i}_mean")
        columnas.append(f"mfcc{i}_var")

    # Crear el DataFrame
    df = pd.DataFrame(datos, columns=columnas)
    return df


# Ruta del directorio base
script_dir = os.path.dirname(os.path.abspath(__file__))
directorio_base = os.path.join(script_dir, "audios02", "train")

# Obtener el DataFrame con las rutas, etiquetas y características
df_rutas_con_caracteristicas = obtener_rutas_y_etiquetas_con_caracteristicas(
    directorio_base
)

# Guardar el DataFrame en CSV
print("\n>> Guardando resultados en archivo CSV...")
csv_path = os.path.join(script_dir, "rutas_etiquetas_plate.csv")
df_rutas_con_caracteristicas.to_csv(csv_path, index=False)

print("[OK] Proceso completado exitosamente")
print(f"    - Archivos procesados: {len(df_rutas_con_caracteristicas)}")
print("    - Archivo generado: rutas_etiquetas_plate.csv")
