# Procesamiento batch con Streamlit

La aplicación pública permite procesar varios registros clínicos desde un archivo CSV, además de
conservar la modalidad de predicción individual. Está dirigida a personal médico, clínico o
técnico capacitado y tiene fines exclusivamente académicos.

## Aplicación publicada

[Abrir la aplicación online y batch](https://heart-disease-leosorio123.streamlit.app/)

## Archivo de entrada

Puede descargar y utilizar el archivo
[`heart_prediction_input_example.csv`](../examples/heart_prediction_input_example.csv). Debe
conservar exactamente estas 13 columnas:

```text
age,sex,chest_pain,rest_bp,chol,fbs,rest_ecg,max_hr,exang,old_peak,slope,ca,thal
```

Cada fila representa un registro. Las variables `fbs` y `exang` aceptan `0`/`1` o
`false`/`true`. Los valores categóricos deben usar los dominios incluidos en el archivo de ejemplo.

## Uso de la modalidad batch

1. Abra la pestaña **Predicción en lote**.
2. Descargue el archivo de entrada de ejemplo si necesita una plantilla.
3. Seleccione **Cargar archivo CSV** y elija el archivo con los registros.
4. Revise la vista previa y la cantidad de filas detectadas.
5. Seleccione **Generar predicciones en lote**.
6. Revise el resumen y la tabla de resultados.
7. Seleccione **Descargar predicciones en CSV** para guardar el resultado.

El procesamiento se detiene antes de llamar al modelo cuando faltan columnas, hay columnas
adicionales, el archivo está vacío o algún valor incumple los tipos, rangos o categorías
documentados. El mensaje identifica la fila problemática usando la numeración visible en el CSV.

## Archivo de salida

El archivo [`heart_predictions_example.csv`](../examples/heart_predictions_example.csv) muestra la
estructura esperada. La salida conserva las variables de entrada y agrega:

- `predicted_class`: clase binaria calculada.
- `prediction_label`: interpretación comprensible de la clase.
- `disease_probability`: probabilidad estimada de enfermedad cardiaca.
- `classification_threshold`: umbral utilizado para clasificar.

Los valores de probabilidad pueden variar si se actualiza la versión del modelo.

## Ejecución local

Desde la raíz del repositorio:

```bash
uv sync
uv run streamlit run app.py
```

## Verificación

```bash
uv run pytest tests/inference
uv run pytest --cov
uv run pre-commit run --all-files
```

La URL pública, las pruebas automatizadas, los archivos de ejemplo y la evidencia visual incluida
en el Pull Request demuestran el funcionamiento de ambas modalidades.
