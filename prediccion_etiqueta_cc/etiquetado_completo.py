import os
import librosa
import numpy as np
import pandas as pd
import librosa.display
from pathlib import Path
from tqdm import tqdm

# Definir directorio base
BASE_DIR = Path(__file__).resolve().parent.parent
PREDICCION_CC_DIR = BASE_DIR / "prediccion_etiqueta_cc"


# Función para extraer características acústicas
def extraer_caracteristicas(audio_path):
    # Cargar el archivo de audio
    y, sr = librosa.load(audio_path, sr=8000)

    # Centroide espectral
    centroide = librosa.feature.spectral_centroid(y=y, sr=sr).mean()

    # Dispersión espectral
    dispersión = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

    # Ataque
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    ataque = onset_env.mean()

    # Decaimiento
    decaimiento = librosa.feature.rms(y=y).mean()

    # HNR (Rapidez de tono) - Versión optimizada
    # Usar una ventana más pequeña para acelerar el cálculo
    hnr = librosa.effects.harmonic(y, margin=2.0)
    hnr_mean = np.mean(hnr)

    # MFCCs (tomando los primeros 5)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=5)
    mfcc_mean = np.mean(mfccs, axis=1)

    # Devolver todas las características como un array
    return [centroide, dispersión, ataque, decaimiento, *mfcc_mean, hnr_mean]


# Función para recorrer directorios y capturar las rutas, etiquetas y características
def obtener_rutas_y_etiquetas_con_caracteristicas(directorio_base):
    datos = []

    # Primero, recopilar todos los archivos a procesar
    archivos_a_procesar = []
    for root, dirs, files in os.walk(directorio_base):
        for file in files:
            if file == "drums_louder.wav":
                archivos_a_procesar.append(os.path.join(root, file))

    print(f"\n>> Archivos encontrados: {len(archivos_a_procesar)}")
    print(">> Extrayendo características acústicas...")

    # Recorrer con barra de progreso
    for audio_path in tqdm(archivos_a_procesar, desc="Procesando audios"):
        # Obtener las etiquetas a partir de las carpetas
        partes_ruta = os.path.relpath(audio_path, directorio_base).split(
            os.sep
        )

        # Encontrar las carpetas relevantes (Placa, Electrode, Current Type)
        try:
            idx_placa = [
                i for i, parte in enumerate(partes_ruta) if "Placa_" in parte
            ][0]
            plate_thickness = partes_ruta[idx_placa]  # 'Placa_12mm'
            electrode = partes_ruta[idx_placa + 1]  # 'E6010', 'E6011', etc.
            current_type = partes_ruta[idx_placa + 2]  # 'AC' o 'DC'

            # Extraer características del archivo de audio
            características = extraer_caracteristicas(audio_path)

            # Agregar fila a la lista de datos
            datos.append(
                [audio_path, plate_thickness, electrode, current_type]
                + características
            )
        except IndexError:
            # Si no se encuentra la estructura esperada, omitir
            continue
        except Exception as e:
            print(f"\n[ERROR] Archivo {audio_path}: {e}")
            continue

    # Crear el DataFrame
    columnas = [
        "Audio Path",
        "Plate Thickness",
        "Electrode",
        "Polarity",
        "Spectral Centroid",
        "Spectral Bandwidth",
        "Onset Attack",
        "RMS Decay",
        "MFCC_1",
        "MFCC_2",
        "MFCC_3",
        "MFCC_4",
        "MFCC_5",
        "HNR",
    ]
    df = pd.DataFrame(datos, columns=columnas)

    return df


# Ruta del directorio base
directorio_base = PREDICCION_CC_DIR / "audios" / "train"

# Obtener el DataFrame
df_rutas_con_caracteristicas = obtener_rutas_y_etiquetas_con_caracteristicas(
    str(directorio_base)
)

# Guardar el DataFrame en CSV
df_rutas_con_caracteristicas.to_csv(
    PREDICCION_CC_DIR / "rutas_etiquetas_completos.csv", index=False
)

print("CSV generado correctamente.")
