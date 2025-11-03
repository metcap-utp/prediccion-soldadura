import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Cargar los datos
print("=" * 60)
print("ENTRENAMIENTO - ELECTRODE")
print("=" * 60)
print("\n>> Cargando datos desde rutas_etiquetas_01.csv...")
file_path = "rutas_etiquetas_01.csv"
data = pd.read_csv(file_path)
print(f"   [OK] {len(data)} muestras cargadas exitosamente")

# Preprocesar las etiquetas
print("\n>> Preprocesando etiquetas con LabelBinarizer...")
label_binarizer = LabelBinarizer()
data["Electrode"] = label_binarizer.fit_transform(data["Electrode"])
print("   [OK] Etiquetas transformadas a formato binario")


# Función para convertir características de listas a valores numéricos
def convertir_caracteristicas_a_numeros(df):
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].apply(
                    lambda x: (
                        np.mean(eval(x))
                        if isinstance(x, str) and x.startswith("[")
                        else float(x)
                    )
                )
            except:
                print(f"Error en la conversión de la columna: {col}")
    return df


# Convertir las características en el DataFrame a valores numéricos
print("\n>> Convirtiendo características a valores numéricos...")
data = convertir_caracteristicas_a_numeros(data)
print("   [OK] Conversión completada")

# Extraer características (X) y etiquetas (y)
print("\n>> Extrayendo características y etiquetas...")
X = data.drop(columns=["Audio Path", "Electrode"]).to_numpy(dtype=np.float32)
y = data[["Electrode"]].to_numpy(dtype=np.float32)
print(f"   [OK] Forma de datos: X={X.shape}, y={y.shape}")

# Dividir los datos en entrenamiento y prueba
print("\n>> Dividiendo datos en conjuntos de entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"   [OK] Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

# Redimensionar las características para redes convolucionales
print("\n>> Redimensionando datos para arquitectura convolucional...")
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
print(f"   [OK] Nueva forma: Train={X_train.shape}, Test={X_test.shape}")

# Crear el modelo de red neuronal
print("\n>> Construyendo arquitectura de red neuronal Conv1D...")
model = keras.Sequential(
    [
        keras.layers.Input(shape=(X.shape[1], 1)),
        keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(
            1, activation="sigmoid"
        ),  # Para clasificación binaria
    ]
)

# Compilar el modelo
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
    ],
)

# Entrenar el modelo
print("\n>> Iniciando proceso de entrenamiento...")
print("   Configuración: 100 épocas, batch size=32, validación=20%")
print("   (Esto puede tomar varios minutos...)\n")
history = model.fit(
    X_train, y_train, epochs=100, batch_size=32, validation_split=0.2
)

# Evaluar el modelo
print("\n>> Evaluando rendimiento del modelo en conjunto de prueba...")
results = model.evaluate(X_test, y_test)
print("\n   Métricas finales:")
print(f"   - Accuracy:  {results[1] * 100:.2f}%")
print(f"   - Precision: {results[2] * 100:.2f}%")
print(f"   - Recall:    {results[3] * 100:.2f}%")

# Guardar el modelo
print("\n>> Guardando modelo entrenado en disco...")
model.save("my_model_completo01.keras")
print("[OK] Entrenamiento finalizado exitosamente")
print("    Modelo guardado como: my_model_completo01.keras")
