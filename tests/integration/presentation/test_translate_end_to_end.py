import re

import pytest
from fastapi.testclient import TestClient

from sentinel_rosetta.infrastructure.opencc_adapter import OpenCcAdapter
from sentinel_rosetta.presentation import cli
from sentinel_rosetta.presentation.controller import app
from tests.example_sentences import EXAMPLE_CHINESE, EXAMPLE_ENGLISH
from tests.model_environment import missing_translation_models

_MISSING = missing_translation_models()

pytestmark = pytest.mark.skipif(
    bool(_MISSING), reason=f"translation models missing: {_MISSING}"
)


def test_cli_real_run_en_to_zh_prints_one_traditional_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Act: no glossary file is read; the service has no term handling.
    exit_code = cli.main(["translate", "-s", "en", "-t", "zh", EXAMPLE_ENGLISH])

    # Assert
    lines = capsys.readouterr().out.strip().splitlines()
    assert exit_code == 0
    assert len(lines) == 1
    assert lines[0] != ""
    assert OpenCcAdapter().to_traditional(lines[0]) == lines[0]


def test_cli_real_run_zh_to_en_prints_one_english_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Act
    exit_code = cli.main(["translate", "-s", "zh", "-t", "en", EXAMPLE_CHINESE])

    # Assert
    lines = capsys.readouterr().out.strip().splitlines()
    assert exit_code == 0
    assert len(lines) == 1
    assert re.search(r"[A-Za-z]", lines[0])


def test_http_real_run_returns_translation_result_fields() -> None:
    # Act
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/translate",
            json={"text": "Hello", "source_lang": "en", "target_lang": "zh"},
        )

    # Assert
    body = response.json()
    assert response.status_code == 200
    assert body["source_text"] == "Hello"
    assert body["target_text"].strip() != ""
