from .abstract_classifier import AbstractClassifier
from .base_ml import BaseML, BASE_MODELS
from .dnn1 import DNN1


def create(method: str, load_pretrained=False, **kwargs) -> AbstractClassifier:
    if method in BASE_MODELS:
        return BaseML(method, load_pretrained, **kwargs)
    elif method == 'dnn1':
        return DNN1(load_pretrained, **kwargs)
    else:
        raise RuntimeError("invalid method " + method)
