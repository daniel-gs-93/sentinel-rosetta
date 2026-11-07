"""Resource-aware: FastAPI routes that serve the translate endpoint."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status

from sentinel_rosetta.application.use_cases import TranslateTextUseCase
from sentinel_rosetta.domain.interfaces import TranslationInputTooLongError
from sentinel_rosetta.domain.models import TranslationResult
from sentinel_rosetta.presentation.bootstrap import (
    build_translate_text_use_case,
)
from sentinel_rosetta.presentation.dtos import TranslationRequest

_logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.translate_text_use_case = build_translate_text_use_case()
    yield


app = FastAPI(title="Sentinel Rosetta", lifespan=lifespan)


def get_translate_text_use_case(request: Request) -> TranslateTextUseCase:
    return request.app.state.translate_text_use_case  # type: ignore[no-any-return]


@app.post("/api/v1/translate", response_model=TranslationResult)
async def translate_text(
    request: TranslationRequest,
    use_case: TranslateTextUseCase = Depends(get_translate_text_use_case),
) -> TranslationResult:
    try:
        return use_case.execute(
            request.text, request.source_lang, request.target_lang
        )
    except TranslationInputTooLongError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(e)
        ) from e
    except Exception as e:
        _logger.exception("Unexpected failure in translate_text")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e
