import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras

# Cargar los datos
print("=" * 60)
print("ENTRENAMIENTO - PLATE THICKNESS")
print("=" * 60)
print("\n>> Cargando datos desde rutas_etiquetas_plate.csv...")
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "..", "rutas_etiquetas_plate.csv")
data = pd.read_csv(file_path)
print(f"   [OK] {len(data)} muestras cargadas exitosamente")

# Preprocesar las etiquetas
print("\n>> Preprocesando etiquetas con LabelBinarizer...")
label_binarizers = {}
for column in ["Plate Thickness"]:
    lb = LabelBinarizer()
    data[column] = lb.fit_transform(data[column])
    label_binarizers[column] = lb

# Convertir las etiquetas a formato numérico adecuado
y = data[["Plate Thickness"]].to_numpy(dtype=np.float32)
print("   [OK] Etiquetas transformadas a formato binario")


# Función para convertir las características de listas a valores numéricos
def convertir_caracteristicas_a_numeros(data):
    for col in data.columns:
        if isinstance(data[col].iloc[0], str) and data[col].iloc[0].startswith("["):
            data[col] = data[col].apply(
                lambda x: np.array(eval(x)) if isinstance(x, str) else x
            )  # Convierte solo si es string
            data[col] = data[col].apply(
                lambda x: x.flatten() if isinstance(x, np.ndarray) else x
            )  # Aplana solo si es array
            data[col] = data[col].apply(
                lambda x: np.mean(x) if isinstance(x, np.ndarray) else x
            )  # Calcula la media si es array
    return data


# Convertir las características en el DataFrame a valores numéricos
print("\n>> Convirtiendo características a valores numéricos...")
data = convertir_caracteristicas_a_numeros(data)
print("   [OK] Conversión completada")

# Extraer las características, excluyendo las etiquetas y la ruta de audio
print("\n>> Extrayendo características y etiquetas...")
X = data.drop(columns=["Audio Path", "Plate Thickness"]).to_numpy(dtype=np.float32)
print(f"   [OK] Forma de datos: X={X.shape}, y={y.shape}")

# Dividir los datos en conjuntos de entrenamiento y prueba
print("\n>> Dividiendo datos en conjuntos de entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"   [OK] Train: {len(X_train)} muestras | Test: {len(X_test)} muestras")

"""
# Verificar que y_train es un ndarray y su forma
print("y_train type:", type(y_train), "Shape:", y_train.shape)
"""
# Redimensionar las características para su uso en redes convolucionales
print("\n>> Redimensionando datos para arquitectura convolucional...")
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]
print(f"   [OK] Nueva forma: Train={X_train.shape}, Test={X_test.shape}")

# Crear el modelo de red neuronal con capas convolucionales
print("\n>> Construyendo arquitectura de red neuronal Conv1D...")
model = keras.Sequential(
    [
        keras.layers.Input(shape=(X.shape[1], 1)),  # Añadir dimensión para el canal
        keras.layers.Conv1D(32, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
        keras.layers.MaxPooling1D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(y.shape[1], activation="sigmoid"),  # Salida multietiqueta
    ]
)

# Compilar el modelo con loss adecuado para clasificación multietiqueta
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",  # Función de pérdida para multietiqueta
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

# Entrenar el modelo
print("\n>> Iniciando proceso de entrenamiento...")
print("   Configuración: 100 épocas, batch size=32, validación=20%")
print("   (Esto puede tomar varios minutos...)\n")
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluar el modelo
print("\n>> Evaluando rendimiento del modelo en conjunto de prueba...")
results = model.evaluate(X_test, y_test)
print("\n   Métricas finales:")
print(f"   - Accuracy:  {results[1] * 100:.2f}%")
print(f"   - Precision: {results[2] * 100:.2f}%")
print(f"   - Recall:    {results[3] * 100:.2f}%")

# Guardar el modelo entrenado
print("\n>> Guardando modelo entrenado en disco...")
uc_dir = os.path.join(script_dir, "..")
model_path = os.path.join(uc_dir, "my_model_completo02.keras")
model.save(model_path)
print("[OK] Entrenamiento finalizado exitosamente")
print("    Modelo guardado como: my_model_completo02.keras")
