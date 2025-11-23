import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Definir directorio base
BASE_DIR = Path(__file__).resolve().parent.parent
PREDICCION_CC_DIR = BASE_DIR / "prediccion_etiqueta_cc"

# Cargar los datos
file_path = PREDICCION_CC_DIR / "rutas_etiquetas_completos.csv"
data = pd.read_csv(file_path)

# Preprocesar las etiquetas
label_binarizers = {}
for column in ["Plate Thickness", "Electrode", "Type of Current"]:
    lb = LabelBinarizer()
    data[column] = lb.fit_transform(data[column])
    label_binarizers[column] = lb

# Convertir las etiquetas a formato numérico adecuado
y = data[["Plate Thickness", "Electrode", "Type of Current"]].to_numpy(
    dtype=np.float32
)

# Extraer las características, excluyendo las etiquetas y la ruta de audio
X = data.drop(
    columns=["Audio Path", "Plate Thickness", "Electrode", "Type of Current"]
).to_numpy(dtype=np.float32)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

"""
# Verificar que y_train es un ndarray y su forma
print("y_train type:", type(y_train), "Shape:", y_train.shape)
"""
# Mostrar información del dataset
print("\n>> Información del dataset:")
print(f"   Total muestras: {len(X)}")
print(f"   Características por muestra: {X.shape[1]}")
print(f"   Conjunto entrenamiento: {len(X_train)} muestras")
print(f"   Conjunto prueba: {len(X_test)} muestras")

# Redimensionar las características para su uso en redes convolucionales
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Crear el modelo de red neuronal con capas convolucionales
# Arquitectura más profunda para aprovechar las 168 características
print("\n>> Creando modelo de red neuronal...")
model = keras.Sequential(
    [
        keras.layers.Input(shape=(X.shape[1], 1)),
        # Primera capa convolucional
        keras.layers.Conv1D(64, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.3),
        # Segunda capa convolucional
        keras.layers.Conv1D(128, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.3),
        # Tercera capa convolucional
        keras.layers.Conv1D(256, 3, activation="relu", padding="same"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),
        keras.layers.Dropout(0.3),
        # Capas densas
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(64, activation="relu"),
        # Salida multietiqueta
        keras.layers.Dense(y.shape[1], activation="sigmoid"),
    ]
)

# Mostrar resumen del modelo
model.summary()

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

# Entrenar el modelo con early stopping
print("\n>> Entrenando modelo...")
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=15, restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1,
)

# Evaluar el modelo
print("\n>> Evaluando modelo en conjunto de prueba...")
results = model.evaluate(X_test, y_test, verbose=0)
print("\n   Métricas finales:")
print(f"   Accuracy:  {results[1] * 100:.2f}%")
print(f"   Precision: {results[2] * 100:.2f}%")
print(f"   Recall:    {results[3] * 100:.2f}%")

# Guardar el modelo entrenado en formato nativo Keras
model_path = PREDICCION_CC_DIR / "my_model_completo.keras"
model.save(model_path)
print(f"\n>> Modelo guardado: {model_path}")
print("   [OK] Entrenamiento completado exitosamente")
