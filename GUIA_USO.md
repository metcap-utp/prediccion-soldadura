# Guía de Uso - Pipeline de Clasificación de Audio de Soldadura

Esta guía explica el orden de ejecución y la función de cada script en el proyecto.

---

## Estructura del Proyecto

El proyecto está organizado en tres directorios principales:

- **`prediccion_etiqueta_cc/`** - Clasificación Completa Combinada (Plate + Electrode + Type of Current)
- **`prediccion_etiqueta_uc/`** - Clasificación por Etiqueta Única (separadas)
- **`prediccion_etiqueta_usg/`** - Clasificación por Segmentos de Tiempo (5s, 10s, 30s)

---

## Predicción Etiqueta CC (Clasificación Completa Combinada)

**Directorio:** `prediccion_etiqueta_cc/`

Este módulo clasifica las tres etiquetas simultáneamente usando un único modelo multi-salida.

### Orden de Ejecución:

#### 1. `etiquetado_completo.py`

**Función:** Extrae características de audio y genera el dataset CSV.

**Qué hace:**

- Escanea recursivamente el directorio `Audios/Train/`
- Extrae 40 características MFCC por audio:
  - 20 coeficientes MFCC (media)
  - 20 coeficientes MFCC (desviación estándar)
- Extrae etiquetas de la estructura de carpetas: `Placa_XXmm/EXXXX/AC_o_DC/`
- Genera el archivo `Rutas_Etiquetas_Completos.csv`

**Salida:**

- `Rutas_Etiquetas_Completos.csv` con rutas y las 40 características MFCC

```bash
python prediccion_etiqueta_cc/etiquetado_completo.py
```

---

#### 2. `entrenamiento_completo.py`

**Función:** Entrena el modelo de clasificación multi-etiqueta.

**Qué hace:**

- Lee el CSV generado en el paso anterior
- Crea un modelo Conv1D con:
  - 3 capas convolucionales
  - BatchNormalization y Dropout
  - 3 salidas independientes (Plate, Electrode, Type of Current)
- Entrena con early stopping
- Guarda el modelo entrenado

**Entrada:**

- `Rutas_Etiquetas_Completos.csv` (con 40 características MFCC)

**Salida:**

- `my_model_completo.keras` (modelo entrenado)

```bash
python prediccion_etiqueta_cc/entrenamiento_completo.py
```

---

#### 3. `prediccion_completo.py`

**Función:** Realiza predicciones sobre audios nuevos.

**Qué hace:**

- Carga el modelo entrenado
- Extrae las mismas 40 características MFCC del audio de prueba
- Predice las tres etiquetas simultáneamente
- Muestra los resultados

**Entrada:**

- `my_model_completo.keras`
- Archivo de audio a predecir

**Salida:**

- Predicción de Plate Thickness, Electrode y Type of Current

```bash
python prediccion_etiqueta_cc/prediccion_completo.py
```

---

## Predicción Etiqueta UC (Clasificación Unitaria)

**Directorio:** `prediccion_etiqueta_uc/`

Este módulo entrena modelos independientes para cada etiqueta por separado.

### Estructura de Subdirectorios:

```
prediccion_etiqueta_uc/
├── audios01/          # Dataset de audio
├── entrenamiento/     # Scripts de entrenamiento
└── prediccion/        # Scripts de predicción
```

### Orden de Ejecución:

#### 1. Etiquetado (Generar CSVs)

**Scripts:**

- `etiquetado_plate.py` - Genera etiquetas de espesor de placa
- `etiquetado_electrode.py` - Genera etiquetas de electrodo
- `etiquetado_current_type.py` - Genera etiquetas de type of current

**Qué hacen:**

- Escanean el directorio `audios01/train/`
- Extraen etiquetas de la estructura de carpetas
- Generan CSVs independientes por etiqueta

**Salidas:**

- `rutas_etiquetas_plate.csv`
- `rutas_etiquetas_electrode.csv`
- `rutas_etiquetas_current_type.csv`

```bash
python prediccion_etiqueta_uc/etiquetado_plate.py
python prediccion_etiqueta_uc/etiquetado_electrode.py
python prediccion_etiqueta_uc/etiquetado_current_type.py
```

---

#### 2. Entrenamiento (Modelos Independientes)

**Directorio:** `entrenamiento/`

**Scripts:**

- `entrenamiento_plate.py` - Entrena modelo de clasificación de placas
- `entrenamiento_electrode.py` - Entrena modelo de clasificación de electrodos
- `entrenamiento_current_type.py` - Entrena modelo de clasificación de type of current

**Qué hacen:**

- Leen sus respectivos CSVs
- Extraen características VGGish de los audios
- Entrenan un modelo Conv1D específico para cada etiqueta
- Guardan los modelos entrenados

**Entradas:**

- CSVs generados en el paso anterior
- Audios en `audios01/train/`

**Salidas:**

- `my_model_plate.keras`
- `my_model_electrode.keras`
- `my_model_current_type.keras`

```bash
python prediccion_etiqueta_uc/entrenamiento/entrenamiento_plate.py
python prediccion_etiqueta_uc/entrenamiento/entrenamiento_electrode.py
python prediccion_etiqueta_uc/entrenamiento/entrenamiento_current_type.py
```

---

#### 3. Predicción (Modelos Independientes)

**Directorio:** `prediccion/`

**Scripts:**

- `prediccion_plate.py` - Predice espesor de placa
- `prediccion_electrode.py` - Predice tipo de electrodo
- `prediccion_current_type.py` - Predice type of current

**Qué hacen:**

- Cargan su modelo correspondiente
- Extraen características VGGish del audio
- Realizan la predicción de su etiqueta específica

**Entradas:**

- Modelos entrenados (`my_model_*.keras`)
- Archivo de audio a predecir

**Salidas:**

- Predicción de la etiqueta correspondiente

```bash
python prediccion_etiqueta_uc/prediccion/prediccion_plate.py
python prediccion_etiqueta_uc/prediccion/prediccion_electrode.py
python prediccion_etiqueta_uc/prediccion/prediccion_current_type.py
```

---

## Predicción Etiqueta USG (Clasificación por Segmentos)

**Directorio:** `prediccion_etiqueta_usg/`

Este módulo clasifica audios segmentados en diferentes duraciones (5s, 10s, 30s).

### Estructura:

```
prediccion_etiqueta_usg/
├── 05s/  # Segmentos de 5 segundos
├── 10s/  # Segmentos de 10 segundos
└── 30s/  # Segmentos de 30 segundos
```

Cada subdirectorio contiene los mismos archivos y sigue el mismo proceso.

### Orden de Ejecución (para cada directorio):

#### 0. Regenerar CSVs (Solo una vez o cuando cambien los audios)

**Script:** `regenerar_csv_usg_local.py` (en la raíz del proyecto)

**Qué hace:**

- Escanea el directorio de audios de entrenamiento
- Genera rutas relativas para portabilidad
- Crea CSVs con las etiquetas para los 3 directorios (05s, 10s, 30s)

**Salidas:**

- `05s/rutas_etiquetas_plate.csv`
- `05s/rutas_etiquetas_electrode.csv`
- `05s/rutas_etiquetas_current_type.csv`
- `10s/rutas_etiquetas_*.csv` (incluye `conjunto.csv`)
- `30s/rutas_etiquetas_*.csv`

```bash
python regenerar_csv_usg_local.py
```

---

#### 1. `training_vggish_completo.py`

**Directorio:** `prediccion_etiqueta_usg/05s/`, `10s/` o `30s/`

**Qué hace:**

- Lee los CSVs con rutas y etiquetas
- Convierte rutas relativas a absolutas
- Carga audios y genera espectrogramas log-mel (formato VGGish)
- Entrena un modelo Conv1D multi-salida
- Guarda el modelo entrenado

**Entradas:**

- `rutas_etiquetas_*.csv`
- Audios en `prediccion_etiqueta_cc/audios/train/`

**Salidas:**

- `my_model_vggish_completo.keras`
- Métricas de entrenamiento (accuracy por etiqueta)

```bash
# Para 5 segundos
python prediccion_etiqueta_usg/05s/training_vggish_completo.py

# Para 10 segundos
python prediccion_etiqueta_usg/10s/training_vggish_completo.py

# Para 30 segundos
python prediccion_etiqueta_usg/30s/training_vggish_completo.py
```

---

#### 2. `modelo_vggish_completo.py`

**Directorio:** `prediccion_etiqueta_usg/05s/`, `10s/` o `30s/`

**Qué hace:**

- Carga el modelo entrenado
- Selecciona aleatoriamente 10 archivos de prueba
- Extrae características VGGish
- Realiza predicciones
- Calcula accuracy por etiqueta y total

**Entradas:**

- `my_model_vggish_completo.keras`
- Audios de prueba en `prediccion_etiqueta_cc/audios/test/`

**Salidas:**

- Predicciones vs etiquetas reales
- Métricas de accuracy

```bash
# Para 5 segundos
python prediccion_etiqueta_usg/05s/modelo_vggish_completo.py

# Para 10 segundos
python prediccion_etiqueta_usg/10s/modelo_vggish_completo.py

# Para 30 segundos
python prediccion_etiqueta_usg/30s/modelo_vggish_completo.py
```

---

## Resumen del Flujo de Trabajo

### Para CC (Clasificación Completa):

```
1. etiquetado_completo.py → CSV con características
2. entrenamiento_completo.py → Modelo entrenado
3. prediccion_completo.py → Predicciones
```

### Para UC (Clasificación Unitaria):

```
1. etiquetado_*.py (x3) → CSVs por etiqueta
2. entrenamiento_*.py (x3) → Modelos por etiqueta
3. prediccion_*.py (x3) → Predicciones por etiqueta
```

### Para USG (Clasificación por Segmentos):

```
0. regenerar_csv_usg_local.py → CSVs para 05s, 10s, 30s
1. training_vggish_completo.py → Modelo por duración
2. modelo_vggish_completo.py → Evaluación y predicciones
```

---

## Recomendaciones

1. **Ejecuta primero CC** - Es el más completo y da mejores resultados
2. **Regenera CSVs de USG** si cambias la estructura de audios
3. **Usa rutas relativas** - Todos los scripts están configurados para portabilidad
4. **Verifica GPU** - Los entrenamientos usan GPU si está disponible
5. **Monitorea métricas** - Cada entrenamiento muestra accuracy por época

---

## Notas Importantes

- **Características por módulo**:
  - **CC**: 40 MFCC (20 mean + 20 std)
  - **UC**: 128 embeddings VGGish
  - **USG**: Log-mel spectrograms (96×64 frames)
- **VGGish**: Modelo pre-entrenado que extrae embeddings de audio a 16kHz (usado en UC)
- **Formato de modelo**: Todos los modelos se guardan en formato `.keras` (nativo de Keras 3.x)
- **Estructura de carpetas**: Las etiquetas se extraen automáticamente de: `Placa_XXmm/EXXXX/AC_o_DC/`

---

## Utilidades

### `regenerar_csv_usg_local.py`

Script de utilidad para regenerar todos los CSVs de USG con rutas relativas actualizadas.

```bash
python regenerar_csv_usg_local.py
```

---

**Última actualización:** Noviembre 12, 2025
