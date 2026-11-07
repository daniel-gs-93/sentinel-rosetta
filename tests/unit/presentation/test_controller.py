from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from sentinel_rosetta.application.use_cases import TranslateTextUseCase
from sentinel_rosetta.domain.interfaces import TranslationInputTooLongError
from sentinel_rosetta.domain.models import LanguageMap, TranslationResult
from sentinel_rosetta.presentation import controller
from sentinel_rosetta.presentation.controller import (
    app,
    get_translate_text_use_case,
)


@pytest.fixture
def translate_text_use_case() -> MagicMock:
    return MagicMock(spec=TranslateTextUseCase)


@pytest.fixture
def client(translate_text_use_case: MagicMock) -> Iterator[TestClient]:
    def get_test_translate_text_use_case() -> MagicMock:
        return translate_text_use_case

    # Overriding the dependency means the real CTranslate2Adapter, and the
    # app's lifespan that constructs it, are never touched by this test.
    app.dependency_overrides[get_translate_text_use_case] = (
        get_test_translate_text_use_case
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_translate_text_valid_request_returns_target_text(
    client: TestClient, translate_text_use_case: MagicMock
) -> None:
    # Arrange
    translate_text_use_case.execute.return_value = TranslationResult(
        source_text="Hello",
        target_text="你好",
        source_lang=LanguageMap.ENGLISH,
        target_lang=LanguageMap.MANDARIN,
    )

    # Act
    response = client.post(
        "/api/v1/translate",
        json={"text": "Hello", "source_lang": "en", "target_lang": "zh"},
    )

    # Assert
    assert response.status_code == 200
    assert response.json() == {
        "source_text": "Hello",
        "target_text": "你好",
        "source_lang": "en",
        "target_lang": "zh",
    }


def test_translate_text_source_equals_target_returns_422(
    client: TestClient, translate_text_use_case: MagicMock
) -> None:
    # Act
    response = client.post(
        "/api/v1/translate",
        json={"text": "Hello", "source_lang": "en", "target_lang": "en"},
    )

    # Assert
    assert response.status_code == 422
    translate_text_use_case.execute.assert_not_called()


def test_translate_text_input_too_long_returns_422(
    client: TestClient, translate_text_use_case: MagicMock
) -> None:
    # Arrange
    translate_text_use_case.execute.side_effect = TranslationInputTooLongError(
        "too long"
    )

    # Act
    response = client.post(
        "/api/v1/translate",
        json={"text": "Hello", "source_lang": "en", "target_lang": "zh"},
    )

    # Assert
    assert response.status_code == 422


def test_translate_text_unexpected_exception_returns_500(
    client: TestClient, translate_text_use_case: MagicMock
) -> None:
    # Arrange
    translate_text_use_case.execute.side_effect = RuntimeError("boom")

    # Act
    response = client.post(
        "/api/v1/translate",
        json={"text": "Hello", "source_lang": "en", "target_lang": "zh"},
    )

    # Assert
    assert response.status_code == 500
    assert "boom" in response.json()["detail"]


def test_lifespan_builds_the_use_case_once_for_many_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    use_case = MagicMock(spec=TranslateTextUseCase)
    use_case.execute.return_value = TranslationResult(
        source_text="Hello",
        target_text="你好",
        source_lang=LanguageMap.ENGLISH,
        target_lang=LanguageMap.MANDARIN,
    )
    build_calls: list[str] = []

    def build_counted_use_case() -> MagicMock:
        build_calls.append("build")
        return use_case

    monkeypatch.setattr(
        controller, "build_translate_text_use_case", build_counted_use_case
    )
    request_body = {"text": "Hello", "source_lang": "en", "target_lang": "zh"}

    # Act
    with TestClient(app) as started_client:
        responses = [
            started_client.post("/api/v1/translate", json=request_body),
            started_client.post("/api/v1/translate", json=request_body),
        ]

    # Assert
    assert len(build_calls) == 1
    assert [response.status_code for response in responses] == [200, 200]


def test_lifespan_startup_failure_propagates_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    def fail_to_build() -> TranslateTextUseCase:
        raise RuntimeError("Model binaries missing for zh")

    monkeypatch.setattr(
        controller, "build_translate_text_use_case", fail_to_build
    )

    # Act / Assert
    with (
        pytest.raises(RuntimeError, match="Model binaries missing"),
        TestClient(app),
    ):
        pass
