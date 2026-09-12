# Feature Pipeline

Este pipeline transforma el archivo RAW del proyecto de enfermedad cardiaca en una tabla de
features reproducible. El proceso:

1. Lee `data/01_raw/corazon.csv` sin inferir tipos ni valores nulos automáticamente.
2. Elimina espacios accidentales y normaliza los tipos definidos durante la exploración.
3. Valida el esquema, los tipos, el porcentaje de nulos, los dominios categóricos y los rangos
   clínicamente plausibles antes de continuar.
4. Elimina registros exactamente duplicados y registros sin una etiqueta válida.
5. Reutiliza `HeartFeatureEngineer` para generar los cuatro atributos derivados definidos en la
   etapa de Feature Engineering.
6. Comprueba la integridad de la tabla final: unicidad de registros, target válido y coherencia de
   cada atributo derivado con sus columnas de origen.
7. Guarda el resultado en `data/04_feature/heart_features.parquet` únicamente cuando todas las
   validaciones son satisfactorias.

La imputación, el escalado y el encoding no se ajustan en este proceso. Esas transformaciones deben
ajustarse exclusivamente con los datos de entrenamiento para evitar fuga de información.

## Contrato de validación

- Se requieren exactamente las 14 columnas documentadas en el archivo fuente.
- Las variables numéricas, booleanas y categóricas deben conservar los tipos anulables definidos
  durante la exploración.
- Ninguna columna puede superar el 10% de valores faltantes después de la corrección estructural.
- `age`, `rest_bp`, `chol`, `max_hr` y `old_peak` deben encontrarse dentro de rangos clínicamente
  plausibles; `slope` debe estar entre 1 y 3 y `ca` entre 0 y 3.
- `sex`, `chest_pain`, `rest_ecg` y `thal` solo aceptan las categorías descritas en el diccionario.
- `fbs`, `exang` y `disease` son variables binarias.
- El archivo fuente no contiene una llave o identificador de paciente. Por eso no es posible
  validar unicidad por paciente; se valida la ausencia de filas completamente duplicadas en la
  tabla final.

Si alguna regla falla, se lanza `DataValidationError` con las columnas y cantidades afectadas. El
archivo Parquet no se crea ni se sobrescribe durante esa ejecución.

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
