import numpy as np
import xgboost
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot as plt
import config
from .abstract_classifier import AbstractClassifier

XGB_MODEL_PATH = config.CACHE_DIR + "/xgb.pth"


class XGBoost(AbstractClassifier):

    def __init__(self, load_pretrained=False, lr=0.1, n_estimators=100, **_kwargs):
        super().__init__()

        if load_pretrained:
            self._model = xgboost.Booster(model_file=XGB_MODEL_PATH)
        else:
            self._model: XGBClassifier = XGBClassifier(max_depth=10, learning_rate=lr, n_estimators=n_estimators,
                                                       use_label_encoder=False)

    def save_to_file(self) -> str:
        self._model.save_model(XGB_MODEL_PATH)
        return XGB_MODEL_PATH

    def _train_inner(self, dataset, show_pic=False, **kwargs) -> np.ndarray:
        self._model.fit(dataset.data, dataset.label)
        if show_pic:
            plot_importance(self._model)
            plt.show()
        return self._model.predict(dataset.data)

    def _test_inner(self, dataset, **kwargs) -> np.ndarray:
        return self._model.predict(dataset.data)
