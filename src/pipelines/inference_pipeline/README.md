# Prediction Pipeline

Este pipeline ejecuta inferencia batch reproducible con el modelo de enfermedad cardiaca generado
por el Training Pipeline. El proceso:

1. Carga el pipeline entrenado desde un archivo `.joblib` del proyecto.
2. Lee un CSV con registros nuevos y las 13 variables clínicas originales, sin incluir el target
   `disease`.
3. Valida el esquema, los tipos, los rangos clínicos y las categorías conocidas.
4. Genera las cuatro variables derivadas mediante `HeartFeatureEngineer`.
5. Comprueba que las 17 variables resultantes coincidan, en nombre y orden, con el esquema del
   modelo almacenado.
6. Aplica la imputación, el escalado y el encoding aprendidos durante el entrenamiento.
7. Calcula la probabilidad de enfermedad cardiaca y la clase correspondiente según el umbral.
8. Conserva los registros de entrada y agrega una etiqueta comprensible, la probabilidad y el
   umbral utilizado.
9. Persiste los resultados en CSV únicamente después de completar el proceso correctamente.

Los valores faltantes se conservan para que sean tratados por la imputación del pipeline. En cambio,
los valores no numéricos, fuera de los rangos definidos o pertenecientes a categorías inválidas
detienen el proceso con un mensaje claro.

## Preparación

Desde la raíz del repositorio, genere primero los features y el modelo:

```bash
uv run python src/pipelines/feature_pipeline/feature_pipeline.py
uv run python src/pipelines/training_pipeline/train_pipeline.py
```

El archivo de entrada debe guardarse, de forma predeterminada, en:

```text
data/05_model_input/heart_prediction_input.csv
```

Columnas requeridas:

```text
age,sex,chest_pain,rest_bp,chol,fbs,rest_ecg,max_hr,exang,old_peak,slope,ca,thal
```

## Ejecución

```bash
uv run python src/pipelines/inference_pipeline/inference_pipeline.py
```

También se pueden indicar rutas y un umbral diferentes:

```bash
uv run python src/pipelines/inference_pipeline/inference_pipeline.py \
  --model ruta/al/modelo.joblib \
  --input ruta/a/registros.csv \
  --output ruta/a/predicciones.csv \
  --threshold 0.5
```

La salida predeterminada se almacena en:

```text
data/07_model_output/heart_predictions.csv
```

Además de las variables originales, contiene:

- `predicted_class`: clase binaria calculada con el umbral configurado.
- `prediction_label`: interpretación comprensible de la clase.
- `disease_probability`: probabilidad estimada de enfermedad cardiaca.
- `classification_threshold`: umbral utilizado.

Este resultado tiene fines exclusivamente académicos. No constituye un diagnóstico, una
recomendación médica ni reemplaza la valoración de profesionales de la salud.

## Pruebas

```bash
uv run pytest tests/pipelines/inference_pipeline
```
