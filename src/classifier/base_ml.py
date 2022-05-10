import numpy as np
import pickle
from sklearn import svm, linear_model, neighbors, gaussian_process, naive_bayes, tree, neural_network, ensemble
from inputs.dataset import Dataset
from .abstract_classifier import AbstractClassifier
from config import BASE_MODELS, CACHE_DIR


class BaseML(AbstractClassifier):
    def __init__(self, model_name: str, load_pretrained=False, n_estimators=10, **_kwargs):
        super().__init__()

        if model_name in BASE_MODELS:
            self._model_path = CACHE_DIR + '/' + model_name + '.pkl'
        else:
            raise Exception("unknown model name")

        if load_pretrained:
            with open(self._model_path, "rb") as f:
                self._model = pickle.loads(f.read())
        else:
            if model_name == 'svm':
                self._model = svm.LinearSVC()
            elif model_name == 'sgd':
                self._model = linear_model.SGDClassifier()
            elif model_name == 'knn':
                self._model = neighbors.KNeighborsClassifier(n_neighbors=10)
            elif model_name == 'gpc':
                self._model = gaussian_process.GaussianProcessClassifier()
            elif model_name == 'bayes':
                self._model = naive_bayes.MultinomialNB()
            elif model_name == 'dt':
                self._model = tree.DecisionTreeClassifier()
            elif model_name == 'mlp':
                self._model = neural_network.MLPClassifier(hidden_layer_sizes=(512, 256, 64, 32))
            elif model_name == 'rf':
                self._model = ensemble.RandomForestClassifier(n_estimators=n_estimators)
            elif model_name == 'gb':
                self._model = ensemble.GradientBoostingClassifier(n_estimators=n_estimators)
            elif model_name == 'ab':
                self._model = ensemble.AdaBoostClassifier(n_estimators=n_estimators)

    def save_to_file(self) -> str:
        with open(self._model_path, "wb") as f:
            f.write(pickle.dumps(self._model))
        return self._model_path

    def _train_inner(self, dataset: Dataset, **kwargs) -> np.ndarray:
        self._model.fit(dataset.data, dataset.label)
        return self._model.predict(dataset.data)

    def _test_inner(self, dataset: Dataset, **kwargs) -> np.ndarray:
        return self._model.predict(dataset.data)
