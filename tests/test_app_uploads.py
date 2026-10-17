from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.customer_ai import FEATURES

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def business_app():
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"),
        default_timeout=45,
    ).run()
    return app.radio[0].set_value("🏢 Business Mode").run()


def test_valid_business_csv_upload_classifies_customers():
    app = business_app()
    sample_path = PROJECT_ROOT / "test_data" / "customer_ai_test_business.csv"

    app.file_uploader[0].upload(
        sample_path.name,
        sample_path.read_bytes(),
    ).run()

    assert not app.exception
    assert app.success
    assert any("10 customers" in message.value for message in app.success)
    assert any("Uploaded Customer Segments" in item.value for item in app.subheader)


@pytest.mark.parametrize(
    "content, message",
    [
        ("", "uploaded CSV is empty"),
        ("Fresh,Milk\n1,2\n", "Missing required columns"),
        (
            ",".join(FEATURES) + "\n1,not-a-number,3,4,5,6\n",
            "non-numeric",
        ),
    ],
)
def test_invalid_business_csv_upload_shows_error(tmp_path, content, message):
    csv_path = tmp_path / "invalid.csv"
    csv_path.write_text(content, encoding="utf-8")
    app = business_app()

    app.file_uploader[0].upload(csv_path.name, csv_path.read_bytes()).run()

    assert not app.exception
    assert app.error
    assert any(message in item.value for item in app.error)
