# Feature Pipeline

Este pipeline transforma el archivo RAW del proyecto de enfermedad cardiaca en una tabla de
features reproducible. El proceso:

1. Lee `data/01_raw/corazon.csv` sin inferir tipos ni valores nulos automáticamente.
2. Elimina espacios accidentales y normaliza los tipos definidos durante la exploración.
3. Elimina registros exactamente duplicados y registros sin una etiqueta válida.
4. Reutiliza `HeartFeatureEngineer` para generar los cuatro atributos derivados definidos en la
   etapa de Feature Engineering.
5. Guarda el resultado en `data/04_feature/heart_features.parquet`.

La imputación, el escalado y el encoding no se ajustan en este proceso. Esas transformaciones deben
ajustarse exclusivamente con los datos de entrenamiento para evitar fuga de información.

## Ejecución

Desde la raíz del repositorio, ubique el archivo entregado para el proyecto en
`data/01_raw/corazon.csv` y ejecute:

```bash
uv sync
uv run python src/pipelines/feature_pipeline/feature_pipeline.py
```

También se pueden indicar rutas diferentes:

```bash
uv run python src/pipelines/feature_pipeline/feature_pipeline.py \
  --input ruta/al/archivo.csv \
  --output ruta/a/features.parquet
```

## Pruebas

```bash
uv run pytest tests/pipelines/feature_pipeline
```
