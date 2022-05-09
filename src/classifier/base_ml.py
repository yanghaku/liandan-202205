import numpy as np
import pickle
from sklearn import svm, linear_model, neighbors, gaussian_process, naive_bayes, tree, neural_network
import config
from inputs.dataset import Dataset
from .abstract_classifier import AbstractClassifier

BASE_MODELS = [
    "svm",
    "sgd",
    "knn",
    "gpc",
    'bayes',
    'dt',
    'mlp',
]


class BaseML(AbstractClassifier):
    def __init__(self, model_name: str, load_pretrained=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if model_name in BASE_MODELS:
            self._model_path = config.CACHE_DIR + '/' + model_name + '.pkl'
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

    def save_to_file(self) -> str:
        with open(self._model_path, "wb") as f:
            f.write(pickle.dumps(self._model))
        return self._model_path

    def _train_inner(self, dataset: Dataset, *args, **kwargs) -> np.ndarray:
        self._model.fit(dataset.data, dataset.label)
        return self._model.predict(dataset.data)

    def _test_inner(self, dataset: Dataset, *args, **kwargs) -> np.ndarray:
        return self._model.predict(dataset.data)
