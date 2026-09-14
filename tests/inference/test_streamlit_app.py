"""Smoke tests for the online and batch Streamlit application."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

EXPECTED_NUMBER_INPUTS = 5
EXPECTED_SELECT_INPUTS = 6
EXPECTED_TABS = 2


def load_app() -> AppTest:
    """Execute the Streamlit app with its default state."""
    project_dir = Path(__file__).resolve().parents[2]
    return AppTest.from_file(str(project_dir / "app.py")).run(timeout=30)


def test_streamlit_app_starts_without_exceptions() -> None:
    """The online demo should render its form and clinical guidance."""
    app = load_app()

    assert not app.exception
    assert app.title[0].value == "❤️ Modelo de enfermedad cardiaca: predicción online y batch"
    assert app.button[0].label == "Generar predicción"
    assert len(app.number_input) == EXPECTED_NUMBER_INPUTS
    assert len(app.selectbox) == EXPECTED_SELECT_INPUTS
    assert len(app.tabs) == EXPECTED_TABS
    assert "personal médico" in app.info[0].value
    assert "no constituye diagnóstico" in app.info[-1].value


def test_streamlit_app_generates_a_prediction() -> None:
    """Submitting the default form should call the persisted pipeline."""
    app = load_app()

    app.button[0].click().run(timeout=30)

    assert not app.exception
    assert len(app.success) + len(app.error) >= 1
    result_message = app.success[0].value if app.success else app.error[0].value
    assert "enfermedad cardiaca" in result_message
    assert app.metric[0].label == "Probabilidad estimada de enfermedad cardiaca"
