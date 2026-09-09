"""Streamlit POC for the selected heart disease model."""

import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from heart_project.prediction import (  # noqa: E402
    PredictivePipeline,
    load_pipeline,
    predict_patient,
)

MODEL_PATH = PROJECT_DIR / "models" / "heart_disease_best_pipeline.joblib"

SEX_LABELS = {"Male": "Masculino", "Female": "Femenino"}
CHEST_PAIN_LABELS = {
    "typical": "Angina típica",
    "asymptomatic": "Asintomático",
    "nonanginal": "Dolor no anginoso",
    "nontypical": "Angina atípica",
}
ECG_LABELS = {
    "normal": "Normal",
    "ST-T wave abnormality": "Anormalidad de la onda ST-T",
    "left ventricular hypertrophy": "Hipertrofia ventricular izquierda",
}
THAL_LABELS = {
    "normal": "Normal",
    "fixed": "Defecto fijo",
    "reversable": "Defecto reversible",
}
SLOPE_LABELS = {1: "1 · Ascendente", 2: "2 · Plana", 3: "3 · Descendente"}


@st.cache_resource(show_spinner="Cargando el modelo entrenado...")
def get_pipeline() -> PredictivePipeline:
    """Load and cache the persisted pipeline."""
    return load_pipeline(MODEL_PATH)


def render_prediction(patient_data: dict[str, object]) -> None:
    """Generate and render one prediction."""
    try:
        result = predict_patient(get_pipeline(), patient_data)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"No fue posible generar la predicción: {error}")
        return

    st.subheader("Resultado del modelo")
    if result.has_disease_pattern:
        st.warning("El modelo identifica un patrón compatible con la clase positiva.")
    else:
        st.success("El modelo identifica un patrón compatible con la clase negativa.")

    probability_column, threshold_column = st.columns(2)
    probability_column.metric(
        "Probabilidad estimada de clase positiva",
        f"{result.disease_probability:.1%}",
    )
    threshold_column.metric("Umbral utilizado", f"{result.threshold:.0%}")
    st.progress(result.disease_probability)

    with st.expander("Ver datos enviados al modelo"):
        st.json(patient_data)


def main() -> None:
    """Render the Streamlit application."""
    st.set_page_config(
        page_title="Demo POC · Riesgo cardiaco",
        page_icon="❤️",
        layout="wide",
    )

    st.title("❤️ Demo POC del modelo de enfermedad cardiaca")
    st.markdown(
        "Esta aplicación local carga el pipeline **Random Forest** seleccionado en la Tarea 6 "
        "y permite comprobar que recibe datos y genera una predicción reproducible."
    )

    with st.sidebar:
        st.header("Información del modelo")
        st.write("**Algoritmo:** Random Forest")
        st.write("**Métrica prioritaria:** sensibilidad (recall)")
        st.write("**Umbral de demostración:** 0,50")
        st.caption("POC académico · No utilizar para decisiones clínicas.")

    st.subheader("Datos de entrada")
    st.caption(
        "Los límites corresponden a los rangos observados en el conjunto de datos del proyecto."
    )

    with st.form("patient_form"):
        demographic, clinical = st.columns(2)

        with demographic:
            st.markdown("#### Información general")
            age = st.number_input("Edad (años)", min_value=29, max_value=77, value=54)
            sex = st.selectbox(
                "Sexo",
                options=list(SEX_LABELS),
                format_func=SEX_LABELS.get,
            )
            chest_pain = st.selectbox(
                "Tipo de dolor torácico",
                options=list(CHEST_PAIN_LABELS),
                format_func=CHEST_PAIN_LABELS.get,
            )
            rest_bp = st.number_input(
                "Presión arterial en reposo (mm Hg)",
                min_value=94,
                max_value=200,
                value=130,
            )
            chol = st.number_input(
                "Colesterol sérico (mg/dl)",
                min_value=126,
                max_value=564,
                value=241,
            )
            fbs = st.checkbox("Glucosa en ayunas > 120 mg/dl")

        with clinical:
            st.markdown("#### Resultados clínicos")
            rest_ecg = st.selectbox(
                "ECG en reposo",
                options=list(ECG_LABELS),
                format_func=ECG_LABELS.get,
            )
            max_hr = st.number_input(
                "Frecuencia cardiaca máxima",
                min_value=71,
                max_value=202,
                value=153,
            )
            exang = st.checkbox("Angina inducida por ejercicio")
            old_peak = st.number_input(
                "Depresión ST inducida por ejercicio",
                min_value=0.0,
                max_value=6.2,
                value=0.8,
                step=0.1,
            )
            slope = st.selectbox(
                "Pendiente del segmento ST",
                options=list(SLOPE_LABELS),
                index=1,
                format_func=SLOPE_LABELS.get,
            )
            ca = st.selectbox("Número de vasos coloreados por fluoroscopia", options=[0, 1, 2, 3])
            thal = st.selectbox(
                "Resultado de thal",
                options=list(THAL_LABELS),
                format_func=THAL_LABELS.get,
            )

        submitted = st.form_submit_button("Generar predicción", type="primary")

    if submitted:
        render_prediction(
            {
                "age": age,
                "sex": sex,
                "chest_pain": chest_pain,
                "rest_bp": rest_bp,
                "chol": chol,
                "fbs": fbs,
                "rest_ecg": rest_ecg,
                "max_hr": max_hr,
                "exang": exang,
                "old_peak": old_peak,
                "slope": slope,
                "ca": ca,
                "thal": thal,
            }
        )

    st.divider()
    st.info(
        "**Aviso:** esta demostración tiene fines exclusivamente académicos. La predicción no "
        "constituye diagnóstico, recomendación médica ni reemplaza la evaluación profesional."
    )


if __name__ == "__main__":
    main()
