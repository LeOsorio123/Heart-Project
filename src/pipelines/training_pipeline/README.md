# Training Pipeline

Este pipeline automatiza el entrenamiento inicial del modelo de enfermedad cardiaca. El proceso:

1. Lee la tabla validada `data/04_feature/heart_features.parquet`.
2. Verifica el esquema, la presencia del target y las clases disponibles.
3. Realiza una separación train/test estratificada, reproducible y sin ajustar transformaciones
   antes de la división.
4. Ajusta la imputación, el escalado y el encoding únicamente con los datos de entrenamiento.
5. Entrena el Random Forest seleccionado en el POC: 250 árboles, `min_samples_leaf=3`, clases
   balanceadas y semilla 42.
6. Evalúa accuracy, recall, especificidad, precisión, F1, balanced accuracy, ROC AUC y matriz de
   confusión. Recall es la métrica prioritaria por el impacto de los falsos negativos.
7. Guarda el pipeline completo y un reporte JSON reproducible.

Los checks avanzados de representatividad y fuga de información del split se incorporan en la
Tarea 4. La validación cruzada y el análisis de generalización corresponden a la Tarea 5.

## Ejecución

Desde la raíz del repositorio:

```bash
uv run python src/pipelines/feature_pipeline/feature_pipeline.py
uv run python src/pipelines/training_pipeline/train_pipeline.py
```

Los artefactos predeterminados se generan en:

- `data/06_models/heart_disease_training_pipeline.joblib`
- `data/07_model_output/training_metrics.json`

También se pueden indicar rutas y parámetros diferentes:

```bash
uv run python src/pipelines/training_pipeline/train_pipeline.py \
  --input ruta/a/heart_features.parquet \
  --model-output ruta/al/modelo.joblib \
  --metrics-output ruta/a/metricas.json \
  --test-size 0.2 \
  --random-state 42
```

## Pruebas

```bash
uv run pytest tests/pipelines/training_pipeline
```
