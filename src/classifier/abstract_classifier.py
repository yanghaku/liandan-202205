import time
import numpy as np
from abc import ABC, abstractmethod
from inputs.dataset import Dataset
from outputs.helper import get_acc, get_f1


class AbstractClassifier(ABC):
    def train(self, dataset: Dataset, save_model=False, **kwargs) -> np.ndarray:
        b = [0, 0, 0, 0, 0, 0]
        for i in dataset.label:
            b[i] += 1
        print("label distribution:", b)

        start = time.time()
        y_pred = self._train_inner(dataset, **kwargs)
        print("train took {:.5f} s".format(time.time() - start))

        if y_pred is not None:
            print("train acc = {:.5f}".format(get_acc(dataset.label, y_pred)))
            print("train macro f1 = {:.5f}".format(get_f1(dataset.label, y_pred)))
        if save_model:
            print("successfully save to " + self.save_to_file())

        return y_pred

    def test(self, dataset: Dataset, **kwargs):
        start = time.time()
        y_pred = self._test_inner(dataset, **kwargs)

        print("test took {:.5f} s".format(time.time() - start))
        if dataset.label is not None:
            print("test acc = {:.5f}".format(get_acc(dataset.label, y_pred)))
            print("test macro f1 = {:.5f}".format(get_f1(dataset.label, y_pred)))
            b = [0, 0, 0, 0, 0, 0]
            for i in dataset.label:
                b[i] += 1
            print("label distribution:", b)
            b = [0, 0, 0, 0, 0, 0]
            for i in y_pred:
                b[i] += 1
            print("predict distribution:", b)

        return y_pred

    @abstractmethod
    def save_to_file(self) -> str:
        pass

    @abstractmethod
    def _train_inner(self, dataset: Dataset, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def _test_inner(self, dataset: Dataset, **kwargs) -> np.ndarray:
        pass
