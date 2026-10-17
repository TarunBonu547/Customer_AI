from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_starts_without_exception():
    app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=15).run()

    assert not app.exception
    assert app.title
