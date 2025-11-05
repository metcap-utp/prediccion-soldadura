"""
Script para regenerar los archivos CSV con rutas relativas a los audios locales de USG.
Los audios están en prediccion_etiqueta_usg/{05s,10s,30s}/audio/train/
"""

from pathlib import Path

import pandas as pd

# Directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent
PREDICCION_USG_DIR = BASE_DIR / "prediccion_etiqueta_usg"


def generar_csv_para_duracion(duracion: str):
    """Genera CSVs para una duración específica (05s, 10s, 30s)."""
    print(f"\n{'=' * 60}")
    print(f"Procesando duración: {duracion}")
    print(f"{'=' * 60}")

    audio_dir = PREDICCION_USG_DIR / duracion / "audio" / "train"

    if not audio_dir.exists():
        print(f"ADVERTENCIA: No existe el directorio {audio_dir}")
        return

    # Listas para almacenar datos
    rutas_plate = []
    rutas_electrode = []
    rutas_polarity = []
    rutas_conjunto = []

    # Buscar todos los archivos .wav recursivamente
    archivos_wav = sorted(audio_dir.rglob("*.wav"))
    print(f"Encontrados {len(archivos_wav)} archivos .wav")

    for archivo_wav in archivos_wav:
        # Obtener ruta relativa desde BASE_DIR
        ruta_relativa = archivo_wav.relative_to(BASE_DIR)

        # Extraer etiquetas de la estructura de carpetas
        # Estructura: prediccion_etiqueta_usg/{duracion}/audio/train/Placa_XXmm/EXXXX/AC_o_DC/...
        partes = archivo_wav.parts

        try:
            # Encontrar índices de las carpetas de etiquetas
            idx_train = partes.index("train")
            placa = partes[idx_train + 1]  # Placa_XXmm
            electrodo = partes[idx_train + 2]  # EXXXX
            polaridad = partes[idx_train + 3]  # AC o DC

            # Validar que las etiquetas tengan el formato esperado
            if not placa.startswith("Placa_"):
                continue
            if not electrodo.startswith("E"):
                continue
            if polaridad not in ["AC", "DC"]:
                continue

            # Agregar a las listas correspondientes
            rutas_plate.append(
                {"Audio Path": str(ruta_relativa), "Plate Thickness": placa}
            )

            rutas_electrode.append(
                {"Audio Path": str(ruta_relativa), "Electrode": electrodo}
            )

            rutas_polarity.append(
                {"Audio Path": str(ruta_relativa), "Polarity": polaridad}
            )

            rutas_conjunto.append(
                {
                    "Audio Path": str(ruta_relativa),
                    "Plate Thickness": placa,
                    "Electrode": electrodo,
                    "Polarity": polaridad,
                }
            )

        except (ValueError, IndexError) as e:
            print(f"ERROR procesando {archivo_wav}: {e}")
            continue

    # Crear DataFrames
    df_plate = pd.DataFrame(rutas_plate)
    df_electrode = pd.DataFrame(rutas_electrode)
    df_polarity = pd.DataFrame(rutas_polarity)
    df_conjunto = pd.DataFrame(rutas_conjunto)

    # Directorio de salida
    output_dir = PREDICCION_USG_DIR / duracion

    # Guardar CSVs
    csv_plate = output_dir / "rutas_etiquetas_plate.csv"
    csv_electrode = output_dir / "rutas_etiquetas_electrode.csv"
    csv_polarity = output_dir / "rutas_etiquetas_polarity.csv"
    csv_conjunto = output_dir / "rutas_etiquetas_conjunto.csv"

    df_plate.to_csv(csv_plate, index=False)
    df_electrode.to_csv(csv_electrode, index=False)
    df_polarity.to_csv(csv_polarity, index=False)
    df_conjunto.to_csv(csv_conjunto, index=False)

    print(f"CSV de plate generado: {csv_plate} ({len(df_plate)} entradas)")
    print(f"CSV de electrode generado: {csv_electrode} ({len(df_electrode)} entradas)")
    print(f"CSV de polarity generado: {csv_polarity} ({len(df_polarity)} entradas)")
    print(f"CSV de conjunto generado: {csv_conjunto} ({len(df_conjunto)} entradas)")

    # Mostrar estadísticas
    print(f"\nEstadisticas para {duracion}:")
    print(f"   Placas: {df_plate['Plate Thickness'].value_counts().to_dict()}")
    print(f"   Electrodos: {df_electrode['Electrode'].value_counts().to_dict()}")
    print(f"   Polaridades: {df_polarity['Polarity'].value_counts().to_dict()}")


def main():
    print("Regenerando archivos CSV para prediccion_etiqueta_usg")
    print(f"Directorio base: {BASE_DIR}")

    # Procesar cada duración
    for duracion in ["05s", "10s", "30s"]:
        generar_csv_para_duracion(duracion)

    print("\n" + "=" * 60)
    print("Regeneracion de CSVs completada")
    print("=" * 60)


if __name__ == "__main__":
    main()
