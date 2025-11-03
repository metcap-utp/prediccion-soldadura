# TODO - Mejoras Pendientes

## 🚀 Alta Prioridad

### Integrar VGGish en prediccion_etiqueta_cc

**Problema actual:**

- El entrenamiento usa 10 características simples de librosa
- La predicción usa VGGish (128 dims) + MFCC (10 dims) = 138 dims → truncadas a 10
- **Inconsistencia entre entrenamiento y predicción**

**Solución propuesta:**

1. **Modificar `etiquetado_completo.py`:**

   - Agregar extracción de características VGGish
   - Opción: Usar solo VGGish (128 dims) o combinar VGGish + características actuales (138 dims)

2. **Modificar `entrenamiento_completo.py`:**

   - Ajustar el modelo para aceptar 128 o 138 características
   - Actualizar la arquitectura de la red según las nuevas dimensiones

3. **Modificar `prediccion_completo.py`:**
   - Eliminar el truncamiento a 10 características
   - Usar las mismas características que en entrenamiento

**Beneficios:**

- ✅ Características más ricas y semánticas
- ✅ Mejor rendimiento del modelo
- ✅ Consistencia total entre entrenamiento y predicción
- ✅ Aprovechamiento del modelo pre-entrenado VGGish

**Archivos a modificar:**

- `prediccion_etiqueta_cc/etiquetado_completo.py`
- `prediccion_etiqueta_cc/entrenamiento_completo.py`
- `prediccion_etiqueta_cc/prediccion_completo.py`

---

## 📊 Media Prioridad

### Optimizaciones adicionales

- [ ] Paralelizar extracción de características con `multiprocessing`
- [ ] Agregar validación cruzada en entrenamiento
- [ ] Implementar early stopping
- [ ] Guardar histórico de entrenamiento (history.pickle)
- [ ] Crear scripts de evaluación de modelos

---

## 🔧 Baja Prioridad

### Refactorización

- [ ] Extraer funciones comunes a un módulo `utils.py`
- [ ] Crear clase base para extractores de características
- [ ] Agregar tests unitarios
- [ ] Documentar funciones con docstrings
- [ ] Crear configuración centralizada (config.yaml)

---

## 📝 Notas

**Estado actual de VGGish:**

- ✅ `prediccion_etiqueta_uc`: Usa VGGish correctamente en predicción
- ❌ `prediccion_etiqueta_cc`: Inconsistencia en el pipeline
- ⚠️ El directorio `vggish_1` contiene el modelo pre-entrenado

**Comandos útiles:**

```bash
# Regenerar etiquetas con VGGish (futuro)
python prediccion_etiqueta_cc/etiquetado_completo.py

# Re-entrenar modelo con nuevas características
python prediccion_etiqueta_cc/entrenamiento_completo.py

# Probar predicción
python prediccion_etiqueta_cc/prediccion_completo.py
```
