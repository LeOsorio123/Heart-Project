"""Smoke tests for the local Streamlit POC."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

EXPECTED_NUMBER_INPUTS = 5
EXPECTED_SELECT_INPUTS = 6


def load_app() -> AppTest:
    """Execute the Streamlit app with its default state."""
    project_dir = Path(__file__).resolve().parents[2]
    return AppTest.from_file(str(project_dir / "app.py")).run(timeout=30)


def test_streamlit_app_starts_without_exceptions() -> None:
    """The POC should render its form and disclaimer."""
    app = load_app()

    assert not app.exception
    assert app.title[0].value == "❤️ Demo POC del modelo de enfermedad cardiaca"
    assert app.button[0].label == "Generar predicción"
    assert len(app.number_input) == EXPECTED_NUMBER_INPUTS
    assert len(app.selectbox) == EXPECTED_SELECT_INPUTS


def test_streamlit_app_generates_a_prediction() -> None:
    """Submitting the default form should call the persisted pipeline."""
    app = load_app()

    app.button[0].click().run(timeout=30)

    assert not app.exception
    assert len(app.success) + len(app.warning) >= 1
    assert app.metric[0].label == "Probabilidad estimada de clase positiva"
    assert app.metric[1].value == "50%"
