# Training Pipeline

Este pipeline automatiza el entrenamiento inicial del modelo de enfermedad cardiaca. El proceso:

1. Lee la tabla validada `data/04_feature/heart_features.parquet`.
2. Verifica el esquema, la presencia del target y las clases disponibles.
3. Realiza una separación train/test estratificada, reproducible y sin ajustar transformaciones
   antes de la división.
4. Valida la separación antes de entrenar: esquema, tamaños, índices compartidos, muestras
   duplicadas, etiquetas y categorías nuevas, además de drift numérico, categórico y del target.
5. Ajusta la imputación, el escalado y el encoding únicamente con los datos de entrenamiento.
6. Entrena el Random Forest seleccionado en el POC: 250 árboles, `min_samples_leaf=3`, clases
   balanceadas y semilla 42.
7. Evalúa accuracy, recall, especificidad, precisión, F1, balanced accuracy, ROC AUC y matriz de
   confusión. Recall es la métrica prioritaria por el impacto de los falsos negativos.
8. Ejecuta `StratifiedKFold` de 5 folds exclusivamente sobre train. En cada fold se ajusta el
   pipeline completo para evitar fuga de información y se registran media, desviación estándar y
   resultados por fold de las métricas principales.
9. Compara train, validación cruzada y el test reservado, y clasifica el comportamiento como
   generalización estable, underfitting, overfitting o generalización inestable.
10. Guarda el pipeline completo y un reporte JSON reproducible que incluye validación del split,
    métricas, comparación, diagnóstico, advertencias y recomendaciones.

Los fallos críticos de integridad detienen el entrenamiento mediante
`TrainTestValidationError`. Las diferencias de distribución se conservan como advertencias para
su revisión porque no prueban por sí solas una fuga de información. Los umbrales predeterminados
son 0.02 para la proporción de test, 0.05 para la tasa positiva y 0.20 para KS y distancia de
variación total.

La validación del modelo usa umbrales explícitos: recall mínimo de 0.75, brecha train-CV máxima de
0.10 y brecha absoluta CV-test máxima de 0.10. Un hallazgo no detiene el pipeline: queda documentado
como advertencia y recomendación porque el test se conserva únicamente para la estimación final y
no se utiliza para ajustar el modelo.

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
