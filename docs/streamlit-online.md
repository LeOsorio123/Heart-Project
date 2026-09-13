# Demo online con Streamlit

La aplicación permite que personal médico, clínico o técnico capacitado diligencie un registro y
obtenga una predicción online del modelo de enfermedad cardiaca. Es una demostración académica y
su resultado no constituye un diagnóstico ni reemplaza la valoración profesional.

## Uso de la aplicación

1. Abra la URL pública de la aplicación.
2. Diligencie los campos usando resultados de exámenes clínicos ya realizados.
3. Seleccione **Generar predicción**.
4. Revise la clasificación y la probabilidad estimada de enfermedad cardiaca.
5. Expanda **Ver detalles técnicos y datos enviados** únicamente si necesita comprobar los valores
   utilizados y el umbral de decisión.

La interfaz no está dirigida a pacientes para autodiagnóstico. Los resultados de ECG, segmento ST,
fluoroscopia y prueba thal deben ser transcritos por personal capacitado.

## Ejecución local

Desde la raíz del repositorio:

```bash
uv sync
uv run streamlit run app.py
```

La aplicación queda disponible normalmente en `http://localhost:8501`. En GitHub Codespaces debe
abrirse el puerto reenviado `8501` desde el panel **Ports**.

## Despliegue en Streamlit Community Cloud

La aplicación se despliega desde este repositorio con la siguiente configuración:

- Repositorio: `LeOsorio123/Heart-Project`
- Rama estable: `main`
- Archivo de entrada: `app.py`
- Versión de Python: `3.12`

Streamlit Community Cloud detecta `uv.lock` en la raíz para instalar las dependencias bloqueadas.
El pipeline entrenado requerido por la aplicación está versionado en
`models/heart_disease_best_pipeline.joblib`.

## Verificación

```bash
uv run pytest tests/inference/test_streamlit_app.py
uv run pytest --cov
uv run pre-commit run --all-files
```

La URL pública y la evidencia visual se incorporan al repositorio después de completar el primer
despliegue.
