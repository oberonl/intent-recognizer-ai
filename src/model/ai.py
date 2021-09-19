from transformers import pipeline

# Модель-классификатор https://huggingface.co/joeddav/xlm-roberta-large-xnli
classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli")


def classify_intent(intent, candidates):
    """Классифицирует intent на одну из категорий из списка candidates, возвращает вероятность принадлежности intent
    к каждой из категорий """
    return classifier(intent, candidates)
