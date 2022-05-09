from .abstract_classifier import AbstractClassifier
from .base_ml import BaseML, BASE_MODELS


def create(method: str, load_pretrained=False, *args, **kwargs) -> AbstractClassifier:
    if method in BASE_MODELS:
        return BaseML(method, load_pretrained, args, kwargs)
