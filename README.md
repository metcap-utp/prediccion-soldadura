# Predicción de Etiquetas de Soldadura con Audio

Este repositorio contiene el código para entrenar y usar modelos de clasificación de audio para predecir características de soldadura.

## Estructura del Proyecto

```
alejandra-2025/
├── prediccion_etiqueta_cc/        # Predicción de etiquetas completas combinadas
│   ├── entrenamiento_completo.py
│   ├── etiquetado_completo.py
│   └── prediccion_completo.py
├── prediccion_etiqueta_uc/        # Predicción de etiquetas individuales
│   ├── entrenamiento/
│   ├── prediccion/
│   └── audios01/
└── diagrams/                      # Diagramas del proyecto
```

## Requisitos

- Python 3.8+
- TensorFlow 2.x
- librosa
- pandas
- numpy
- scikit-learn
- tensorflow-hub

## Uso

### Predicción de Etiquetas Completas (CC)

1. Etiquetar audios:

   ```bash
   python prediccion_etiqueta_cc/etiquetado_completo.py
   ```

2. Entrenar modelo:

   ```bash
   python prediccion_etiqueta_cc/entrenamiento_completo.py
   ```

3. Realizar predicciones:
   ```bash
   python prediccion_etiqueta_cc/prediccion_completo.py
   ```

### Predicción de Etiquetas Individuales (UC)

1. Entrenar modelos individuales:

   ```bash
   python prediccion_etiqueta_uc/entrenamiento/entrenamiento_electrode.py
   python prediccion_etiqueta_uc/entrenamiento/entrenamiento_plate.py
   python prediccion_etiqueta_uc/entrenamiento/entrenamiento_polarity.py
   ```

2. Realizar predicciones:
   ```bash
   python prediccion_etiqueta_uc/prediccion/prediccion_electrode.py
   python prediccion_etiqueta_uc/prediccion/prediccion_plate.py
   python prediccion_etiqueta_uc/prediccion/prediccion_polarity.py
   ```

## Notas

- Todos los scripts utilizan rutas absolutas desde el directorio base `alejandra-2025`
- Los archivos CSV y modelos `.h5` están excluidos del control de versiones
- Los audios de entrenamiento están organizados por tipo de placa, electrodo y polaridad
