import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

# Definir directorio base
BASE_DIR = Path(__file__).resolve().parent.parent
PREDICCION_CC_DIR = BASE_DIR / 'prediccion_etiqueta_cc'

# Cargar los datos
file_path = PREDICCION_CC_DIR / 'rutas_etiquetas_completos.csv'
data = pd.read_csv(file_path)

# Preprocesar las etiquetas
label_binarizers = {}
for column in ['Plate Thickness', 'Electrode', 'Polarity']:
    lb = LabelBinarizer()
    data[column] = lb.fit_transform(data[column])
    label_binarizers[column] = lb

# Convertir las etiquetas a formato numérico adecuado
y = data[['Plate Thickness', 'Electrode', 'Polarity']].to_numpy(dtype=np.float32)

# Extraer las características, excluyendo las etiquetas y la ruta de audio
X = data.drop(columns=['Audio Path', 'Plate Thickness', 'Electrode', 'Polarity']).to_numpy(dtype=np.float32)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

'''
# Verificar que y_train es un ndarray y su forma
print("y_train type:", type(y_train), "Shape:", y_train.shape)
'''
# Redimensionar las características para su uso en redes convolucionales
X_train = X_train[..., np.newaxis]  
X_test = X_test[..., np.newaxis]

# Crear el modelo de red neuronal con capas convolucionales
model = keras.Sequential([
    keras.layers.Input(shape=(X.shape[1], 1)),  # Añadir dimensión para el canal
    keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling1D(2),
    keras.layers.Conv1D(64, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling1D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(y.shape[1], activation='sigmoid')  # Salida multietiqueta
])

# Compilar el modelo con loss adecuado para clasificación multietiqueta
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Función de pérdida para multietiqueta
              metrics=['accuracy', 
                       tf.keras.metrics.Precision(name='precision'), 
                       tf.keras.metrics.Recall(name='recall')])

# Entrenar el modelo
history = model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2
)

# Evaluar el modelo
results = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {results[1] * 100:.2f}%')
print(f'Test Precision: {results[2]:.2f}')
print(f'Test Recall: {results[3]:.2f}')

# Guardar el modelo entrenado
model.save(PREDICCION_CC_DIR / 'my_model_completo.h5')