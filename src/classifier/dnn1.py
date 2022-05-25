import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import config
from .abstract_classifier import AbstractClassifier
from outputs.helper import get_acc, get_f1


class NET(nn.Module):
    def __init__(self, n_feature):
        super(NET, self).__init__()
        self._net = nn.Sequential(
            nn.Linear(n_feature, 1024),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(1024, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self._net(x)


DNN1_MODEL_PATH = config.CACHE_DIR + "/dnn1.pth"


class DNN1(AbstractClassifier):
    def __init__(self, load_pretrained=False, batch=16, lr=0.001, **kwargs):
        super().__init__()

        if load_pretrained:
            self._net = torch.load(DNN1_MODEL_PATH)
        else:
            if 'n_features' in kwargs:
                n_features = int(kwargs['n_features'])
            else:
                raise TypeError("need 'n_features' in params")
            self._net = NET(n_features).double()

        self._loss_func = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)
        self._batch = batch
        self._use_cuda = config.USE_CUDA

        print("USE_CUDA =", self._use_cuda)
        if self._use_cuda:
            self._net = self._net.cuda()
            self._loss_func = self._loss_func.cuda()

    def save_to_file(self) -> str:
        torch.save(self._net, DNN1_MODEL_PATH)
        return DNN1_MODEL_PATH

    def _train_inner(self, dataset, epoch=100, test_data=None, show_pic=False, **kwargs):
        self._net.train()
        shape = dataset.data.shape
        X = torch.from_numpy(dataset.data)
        Y = torch.from_numpy(dataset.label)
        if self._use_cuda:
            X = X.cuda()
            Y = Y.cuda()

        dataloader = DataLoader(dataset=TensorDataset(X, Y), batch_size=self._batch, shuffle=True)

        loss_lst = []
        train_acc_lst = []
        test_acc_lst = []
        test_f1_lst = []

        for e in range(epoch):
            sum_loss = 0.0
            batch_num = 0
            correct_num = 0
            for batch_x, batch_y in dataloader:
                self._optimizer.zero_grad()
                output = self._net(batch_x)
                loss = self._loss_func(output, batch_y)
                loss.backward()
                self._optimizer.step()

                sum_loss += loss.detach().cpu().item()
                batch_num += 1
                _, pred = torch.max(output.data, dim=1)
                correct_num += (pred == batch_y).sum().item()

            avg_loss = sum_loss / batch_num
            train_acc = float(correct_num) / shape[0]
            loss_lst.append(avg_loss)
            train_acc_lst.append(train_acc)
            print("[epoch {}/{}] average_loss={:.5f}  train_acc={:.5f} "
                  .format(e + 1, epoch, avg_loss, train_acc), end=' ')

            if test_data is not None and test_data.label is not None:
                test_pred = self._test_inner(test_data)
                test_acc = get_acc(test_data.label, test_pred)
                test_f1 = get_f1(test_data.label, test_pred)
                test_acc_lst.append(test_acc)
                test_f1_lst.append(test_f1)
                print("test_acc={:.5f}  test_f1={:.5f}".format(test_acc, test_f1))
            else:
                print()

        if show_pic:
            plt.plot(range(len(loss_lst)), loss_lst, label="loss")
            plt.plot(range(len(train_acc_lst)), train_acc_lst, label="train acc")
            if len(test_acc_lst) != 0:
                plt.plot(range(len(test_acc_lst)), test_acc_lst, label="test acc")
                plt.plot(range(len(test_f1_lst)), test_f1_lst, label="test macro f1")
            plt.legend(loc='best')
            if show_pic:
                plt.show()

        return None

    def _test_inner(self, dataset, **kwargs) -> np.ndarray:
        self._net.eval()
        X = torch.from_numpy(dataset.data)
        if self._use_cuda:
            X = X.cuda()
        _, pred = torch.max(self._net(X).data, dim=1)
        return pred.detach().cpu().numpy()
