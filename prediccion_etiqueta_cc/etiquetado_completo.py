import os
from pathlib import Path

import librosa
import librosa.display
import numpy as np
import pandas as pd
from tqdm import tqdm

# Definir directorio base
BASE_DIR = Path(__file__).resolve().parent.parent
PREDICCION_CC_DIR = BASE_DIR / "prediccion_etiqueta_cc"


# Función para extraer características acústicas (solo MFCC)
def extraer_caracteristicas(audio_path):
    try:
        # Cargar el archivo de audio
        y, sr = librosa.load(audio_path, sr=16000)

        # MFCCs (tomando 20 para más información)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Combinar media y desviación estándar (40 características)
        features = np.concatenate([mfcc_mean, mfcc_std])

        # Verificar dimensión final
        assert features.shape[0] == 40, (
            f"Esperadas 40 características, obtenidas: {features.shape[0]}"
        )

        return features.tolist()
    except Exception as e:
        print(f"\n[ERROR] en {audio_path}: {e}")
        return np.zeros(40).tolist()


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
        partes_ruta = os.path.relpath(audio_path, directorio_base).split(os.sep)

        # Encontrar las carpetas relevantes (Placa, Electrode, Current Type)
        try:
            idx_placa = [i for i, parte in enumerate(partes_ruta) if "Placa_" in parte][
                0
            ]
            plate_thickness = partes_ruta[idx_placa]  # 'Placa_12mm'
            electrode = partes_ruta[idx_placa + 1]  # 'E6010', 'E6011', etc.
            current_type = partes_ruta[idx_placa + 2]  # 'AC' o 'DC'

            # Extraer características del archivo de audio
            características = extraer_caracteristicas(audio_path)

            # Agregar fila a la lista de datos
            datos.append(
                [audio_path, plate_thickness, electrode, current_type] + características
            )
        except IndexError:
            # Si no se encuentra la estructura esperada, omitir
            continue
        except Exception as e:
            print(f"\n[ERROR] Archivo {audio_path}: {e}")
            continue

    # Crear el DataFrame
    # Columnas: Audio Path, 3 etiquetas, 40 características
    columnas = ["Audio Path", "Plate Thickness", "Electrode", "Type of Current"]

    # Agregar nombres de características MFCC (40: 20 mean + 20 std)
    columnas += [f"MFCC_{i + 1}_mean" for i in range(20)]
    columnas += [f"MFCC_{i + 1}_std" for i in range(20)]

    df = pd.DataFrame(datos, columns=columnas)

    return df


# Ruta del directorio base
directorio_base = PREDICCION_CC_DIR / "audios" / "train"

# Obtener el DataFrame
df_rutas_con_caracteristicas = obtener_rutas_y_etiquetas_con_caracteristicas(
    str(directorio_base)
)

# Guardar el DataFrame en CSV
output_path = PREDICCION_CC_DIR / "rutas_etiquetas_completos.csv"
df_rutas_con_caracteristicas.to_csv(output_path, index=False)

print("\n>> CSV generado correctamente")
print(f"   Archivo: {output_path}")
print(f"   Muestras: {len(df_rutas_con_caracteristicas)}")
print("   Características por muestra: 40 (MFCC)")
