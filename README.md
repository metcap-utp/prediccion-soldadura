# Clasificación de Audio de Soldadura con Deep Learning

Sistema de clasificación multi-etiqueta para predecir características de soldadura (espesor de placa, tipo de electrodo y type of current) a partir de señales de audio usando modelos de Deep Learning con VGGish.

## Descripción

Este proyecto implementa tres enfoques diferentes para la clasificación de audio de soldadura:

- **CC (Clasificación Completa)**: Modelo multi-salida que predice las 3 etiquetas simultáneamente usando MFCC
- **UC (Clasificación Unitaria)**: Modelos independientes para cada etiqueta usando embeddings VGGish
- **USG (Clasificación por Segmentos)**: Modelos entrenados con diferentes duraciones de audio (5s, 10s, 30s) usando log-mel spectrograms

## Inicio Rápido

### Instalación

```bash
# Clonar repositorio
git clone <url-del-repo>
cd prediccion-soldadura

# Opción 1: Usar conda (recomendado)
conda env create -f environment.yaml
conda activate audio

# Opción 2: Usar pip
pip install -r requirements.txt
```

### Obtener los Audios

Los archivos de audio **no están incluidos** en el repositorio. Descárgalos desde:

- **5s y 10s**: [Audios_Segmentados](https://github.com/AleAvila09/Audios_Segmentados)
- **30s**: [Audios_Segmentado30s](https://github.com/AleAvila09/Audios_Segmentado30s)

Estructura esperada para USG:

```
prediccion_etiqueta_usg/
├── 05s/audio/
│   ├── train/
│   │   ├── Placa_3mm/E6010/AC/.../*.wav
│   │   ├── Placa_3mm/E6010/DC/.../*.wav
│   │   └── ... (más combinaciones)
│   └── test/
├── 10s/audio/
│   └── (misma estructura)
└── 30s/audio/
    └── (misma estructura)
```

Después de descargar los audios, regenera los CSVs:

```bash
python regenerar_csv_usg_local.py
```

### Uso Básico

```bash
# 1. Extraer características y generar dataset
python prediccion_etiqueta_cc/etiquetado_completo.py

# 2. Entrenar modelo
python prediccion_etiqueta_cc/entrenamiento_completo.py

# 3. Realizar predicciones
python prediccion_etiqueta_cc/prediccion_completo.py
```

**Para instrucciones detalladas, consulta [GUIA_USO.md](GUIA_USO.md)**

## Estructura del Proyecto

```
alejandra-2025/
├── prediccion_etiqueta_cc/          # Clasificación Completa Combinada
│   ├── etiquetado_completo.py       # Extracción de características (MFCC)
│   ├── entrenamiento_completo.py    # Entrenamiento del modelo multi-salida
│   ├── prediccion_completo.py       # Predicción de nuevos audios
│   └── Audios/                      # Dataset de audio
│       ├── Train/                   # Datos de entrenamiento
│       └── Test/                    # Datos de prueba
│
├── prediccion_etiqueta_uc/          # Clasificación Unitaria
│   ├── entrenamiento/               # Scripts de entrenamiento por etiqueta
│   │   ├── entrenamiento_plate.py
│   │   ├── entrenamiento_electrode.py
│   │   └── entrenamiento_current_type.py
│   ├── prediccion/                  # Scripts de predicción por etiqueta
│   │   ├── prediccion_plate.py
│   │   ├── prediccion_electrode.py
│   │   └── prediccion_current_type.py
│   └── audios01/                    # Dataset de audio
│       ├── train/
│       └── test/
│
├── prediccion_etiqueta_usg/         # Clasificación por Segmentos
│   ├── 05s/                         # Modelos para segmentos de 5 segundos
│   │   ├── audio/                   # Dataset local (3071 archivos)
│   │   │   ├── train/
│   │   │   └── test/
│   │   ├── training_vggish_completo.py
│   │   ├── modelo_vggish_completo.py
│   │   └── rutas_etiquetas_*.csv    # CSVs generados automáticamente
│   ├── 10s/                         # Modelos para segmentos de 10 segundos (1495 archivos)
│   └── 30s/                         # Modelos para segmentos de 30 segundos (602 archivos)
│
├── diagrams/                        # Diagramas y visualizaciones
├── regenerar_csv_usg_local.py      # Utilidad para regenerar CSVs con audios locales
├── environment.yaml                # Conda environment (audio)
├── requirements.txt                # Dependencias pip
├── GUIA_USO.md                     # Guía detallada de uso
└── README.md                        # Este archivo
```

## Características Técnicas

### Extracción de Características

- **CC (Clasificación Completa)**: 40 coeficientes MFCC (20 mean + 20 std)
- **UC (Clasificación Unitaria)**: 128 embeddings VGGish pre-entrenados
- **USG (Por Segmentos)**: Log-mel spectrograms (96×64) para VGGish

### Arquitectura del Modelo

- **Red Neuronal**: Conv1D con capas de BatchNormalization y Dropout
- **Salidas**: 3 clasificadores independientes (Plate, Electrode, Type of Current)
- **Optimizador**: Adam
- **Función de pérdida**: Sparse Categorical Crossentropy

### Etiquetas Clasificadas

- **Plate Thickness**: Placa_3mm, Placa_6mm, Placa_12mm
- **Electrode**: E6010, E6011, E6013, E7018
- **Type of Current**: AC, DC

## Utilidades

### Regenerar CSVs de USG

Después de descargar o modificar los audios locales, regenera los CSVs:

```bash
python regenerar_csv_usg_local.py
```

Este script:

- Escanea `prediccion_etiqueta_usg/{05s,10s,30s}/audio/train/` recursivamente
- Extrae etiquetas de la estructura de carpetas (`Placa_XXmm/EXXXX/AC_o_DC/`)
- Genera rutas relativas desde el directorio base del proyecto
- Crea CSVs actualizados: `rutas_etiquetas_plate.csv`, `electrode.csv`, `current_type.csv`, `conjunto.csv`
- Muestra estadísticas de distribución por etiqueta
