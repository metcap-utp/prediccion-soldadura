import os
import librosa
import numpy as np
import pandas as pd
import librosa.display
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub

# Definir directorio base
BASE_DIR = Path(__file__).resolve().parent.parent
PREDICCION_CC_DIR = BASE_DIR / "prediccion_etiqueta_cc"

# Cargar el modelo VGGish
print(">> Cargando modelo VGGish...")
vggish_model_handle = str(PREDICCION_CC_DIR / "vggish_1")
vggish_model = hub.load(vggish_model_handle)
print("   [OK] VGGish cargado")


# Función para extraer características con VGGish
def extraer_caracteristicas_vggish(audio_path):
    try:
        # Cargar audio a 16kHz (requerido por VGGish)
        y_audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # VGGish espera entrada 1D sin reshape
        # El modelo procesa la señal completa y retorna embeddings por ventana
        vggish_features = vggish_model(y_audio)

        # Promediar sobre todas las ventanas temporales para obtener 128 características
        vggish_avg = np.mean(vggish_features.numpy(), axis=0)

        # Verificar que tenemos exactamente 128 dimensiones
        assert (
            vggish_avg.shape[0] == 128
        ), f"VGGish debe retornar 128 dims, obtenido: {vggish_avg.shape[0]}"

        return vggish_avg
    except Exception as e:
        print(f"\n[ERROR] VGGish en {audio_path}: {e}")
        # Retornar vector de ceros en caso de error
        return np.zeros(128)


# Función para extraer características acústicas complementarias
def extraer_caracteristicas_librosa(audio_path):
    try:
        # Cargar el archivo de audio
        y, sr = librosa.load(audio_path, sr=16000)

        # MFCCs (tomando 20 para más información)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # Combinar media y desviación estándar (40 características)
        features = np.concatenate([mfcc_mean, mfcc_std])
        return features
    except Exception as e:
        print(f"\n[ERROR] Librosa en {audio_path}: {e}")
        return np.zeros(40)


# Función combinada para extraer todas las características
def extraer_caracteristicas(audio_path):
    # VGGish: 128 características
    vggish_features = extraer_caracteristicas_vggish(audio_path)

    # Librosa: 40 características (20 MFCCs x 2: mean + std)
    librosa_features = extraer_caracteristicas_librosa(audio_path)

    # Combinar: Total 168 características
    combined = np.concatenate([vggish_features, librosa_features])

    # Verificar dimensión final
    assert (
        combined.shape[0] == 168
    ), f"Esperadas 168 características, obtenidas: {combined.shape[0]}"

    return combined.tolist()


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
    # Columnas: Audio Path, 3 etiquetas, 168 características
    columnas = ["Audio Path", "Plate Thickness", "Electrode", "Polarity"]

    # Agregar nombres de características VGGish (128)
    columnas += [f"VGGish_{i+1}" for i in range(128)]

    # Agregar nombres de características MFCC (40: 20 mean + 20 std)
    columnas += [f"MFCC_{i+1}_mean" for i in range(20)]
    columnas += [f"MFCC_{i+1}_std" for i in range(20)]

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

print(f"\n>> CSV generado correctamente")
print(f"   Archivo: {output_path}")
print(f"   Muestras: {len(df_rutas_con_caracteristicas)}")
print(f"   Características por muestra: 168 (128 VGGish + 40 MFCC)")
