import uvicorn
import sys

from typing import List
from loguru import logger

from fastapi import FastAPI
from pydantic import BaseModel, Field

from model.ai import classify_intent

# Текущая версия компонента
VERSION = "1.0.0"


class Request(BaseModel):
    """ДТО-обьект внешнего API компонента """
    session_id: str = Field(None, title="Идентификатор сессии", example="Test")
    intent: str = Field(None, title="Входная последовательность, которую нужно классифицировать", example="Дождь")
    candidates: List[str] = Field(None, title="Категории для классификации", example=["Погода", "Политика", "Спорт"])


class ClassificationResult(BaseModel):
    """Результат отработки модели, классификация входной фразы"""
    sequence: str = Field(None, title="Фраза, классификация которой осуществлялась", example="Дождь")
    labels: List[str] = Field(None, title="Категории для классификации", example=["Погода", "Политика", "Спорт"])
    scores: List[float] = Field(None, title="Вероятность принадлежности входной фразы к категории", example=[0.9, 0.05, 0.05])


class Response(BaseModel):
    session_id: str = Field(None, title="Идентификатор сессии", example="Test")
    classification_result: ClassificationResult = Field(None, title="Результат классификации", example={})
    version: str = Field(None, title="Версия используемого классификатора", example="1.0.0")


logger.remove()
logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | {level} | <level>{message}</level>")


tags_metadata = [
    {
        "name": "intent",
        "description": "Intent classification",
    }
]
app = FastAPI(
    title="ClassificationApp",
    description="Intent classification, no retraining required if you add some new categories into the request",
    version="1.0.0",
    openapi_tags=tags_metadata
)


@app.post("/intent", response_model=Response, tags=["intent"])
async def classify_intent_request(request: Request):
    """Внешний API приложения. При получении запроса вызывает классификатор и возвращает результат классификации обратно """

    logger.info("[{}] incoming classification request {}", request.session_id, request)

    classification_result = classify_intent(request.intent, request.candidates)
    logger.info("[{}] classification result {}", request.session_id, classification_result)

    result = {"session_id": request.session_id,
              "classification_result": classification_result,
              "version": VERSION}

    return result


logger.add(sys.stdout, colorize=True, format="<green>{time:HH:mm:ss}</green> | {level} | <level>{message}</level>")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
