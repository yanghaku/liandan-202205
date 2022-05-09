from .matrix_data import MatrixData
import numpy as np


class Dataset:
    def __init__(self, filename=None, has_header=True, with_normalize=True, matrix_data=None, data=None, label=None):
        # 直接通过data,label构造
        if data is not None:
            self.data = data
            self.label = label
            self.matrix_data = matrix_data
        # 通过指定初始化好的matrix_data 构造
        elif matrix_data is not None:
            self.matrix_data = matrix_data
            self.data, self.label = self.load_data_from_matrix(matrix_data)
        # 通过加载csv文件构造
        elif filename is not None:
            self.matrix_data = MatrixData(filename, has_header, with_normalize)
            self.data, self.label = self.load_data_from_matrix(self.matrix_data)
        else:
            raise Exception("Cannot construct Dataset with params")

    @staticmethod
    def load_data_from_matrix(matrix_data: MatrixData):
        data = np.concatenate((matrix_data.base, matrix_data.review, matrix_data.types,
                               matrix_data.neighbourhood, matrix_data.amenities), axis=1)
        label = matrix_data.label
        return data, label

    # 划分成训练集和测试集
    # percent 为训练集所占的百分比
    def divide_to_train_test(self, percent: float, with_random=False):
        assert 0.0 < percent <= 1.0
        if self.label is None:
            raise Exception("None label cannot be train set")

        if with_random:
            r = np.random.permutation(len(self.data))
            data = self.data[r, :]
            label = self.label[r]
        else:
            data = self.data
            label = self.label
        train_num = int(len(data) * percent)
        train = data[:train_num, :]
        test = data[train_num:, :]
        label_train = label[:train_num]
        label_test = label[train_num:]

        print("success to divide. train shape={}, test shape={}".format(train.shape, test.shape))

        return Dataset(matrix_data=self.matrix_data, data=train, label=label_train), Dataset(
            matrix_data=self.matrix_data, data=test, label=label_test)
