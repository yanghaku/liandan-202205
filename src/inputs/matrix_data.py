from typing import List
import numpy as np
from . import raw_data, encoder
from sklearn.preprocessing import MinMaxScaler


class MatrixData:
    def __init__(self, filename: str, has_header=True, with_normalize=False):
        raw = raw_data.load(filename, has_header)

        # 包含的列: accommodates,bathrooms,bathroom_shared,bedrooms,instant_bookable
        # 经纬度暂不加入('latitude', 'longitude')
        self.base = self.__init_base_matrix(raw)

        # 包含的列 'reviews', 'review_rating', 'review_scores_A', 'review_scores_B', 'review_scores_C', 'review_scores_D'
        self.review = self.__init_review_matrix(raw)

        # onehot之后的矩阵
        self.types = self.__init_types_matrix(raw)
        # onehot之后的矩阵
        self.neighbourhood = self.__init_neighbourhood_matrix(raw)
        # onehot之后的矩阵
        self.amenities = self.__init_amenities_matrix(raw)
        # 语义的处理
        self.description = self.__init_description_matrix(raw)

        # label (测试集可以是none)
        if raw[0].target is None:
            self.label = None
        else:
            self.label = np.zeros((len(raw, ))).astype(np.longlong)
            for i in range(len(raw)):
                self.label[i] = raw[i].target
        if with_normalize:
            self.normalize()

    @staticmethod
    def __init_base_matrix(raw: List[raw_data.RawData]) -> np.ndarray:
        ans = np.zeros((len(raw), 5)).astype(np.float64)
        for i in range(len(raw)):
            ans[i][0] = float(raw[i].accommodates)
            ans[i][1] = raw[i].bathrooms
            ans[i][2] = raw[i].bathroom_shared
            ans[i][3] = raw[i].bedrooms
            ans[i][4] = raw[i].instant_bookable
        return ans

    @staticmethod
    def __init_review_matrix(raw: List[raw_data.RawData]) -> np.ndarray:
        ans = np.zeros((len(raw), 6)).astype(np.float64)
        for i in range(len(raw)):
            # 当前策略是将空值直接置为0
            if raw[i].review_info is None:
                continue
            ans[i][0] = raw[i].review_info.num
            ans[i][1] = raw[i].review_info.rating
            ans[i][2] = raw[i].review_info.score_a
            ans[i][3] = raw[i].review_info.score_b
            ans[i][4] = raw[i].review_info.score_c
            ans[i][5] = raw[i].review_info.score_d
        return ans

    @staticmethod
    def __init_types_matrix(raw: List[raw_data.RawData]) -> np.ndarray:
        lst = []
        for i in raw:
            lst.append([i.type])
        return encoder.TypesEncoder().transform(lst)

    @staticmethod
    def __init_neighbourhood_matrix(raw: List[raw_data.RawData]) -> np.ndarray:
        lst = []
        for i in raw:
            lst.append([i.neighbourhood])
        return encoder.NeighbourhoodEncoder().transform(lst)

    @staticmethod
    def __init_amenities_matrix(raw: List[raw_data.RawData]) -> np.ndarray:
        lst = []
        for i in raw:
            lst.append(i.amenities)
        return encoder.AmenitiesEncoder().transform(lst)

    @staticmethod
    def __init_description_matrix(raw: List[raw_data.RawData]) -> np.ndarray:
        # todo
        pass

    @staticmethod
    def __print_info_inner(matrix: np.ndarray, name: str):
        format_str = "[{}]:\tshape={}, sum={}, max={}, min={}"
        if matrix is None:
            print('[{}]:\tNone'.format(name))
        else:
            print(format_str.format(name, matrix.shape, matrix.sum(), np.max(matrix), np.min(matrix)))

    def print_info(self):
        self.__print_info_inner(self.base, 'base')
        self.__print_info_inner(self.review, 'review')
        self.__print_info_inner(self.types, 'types')
        self.__print_info_inner(self.neighbourhood, 'neighbourhood')
        self.__print_info_inner(self.amenities, 'amenities')
        self.__print_info_inner(self.description, 'description')
        self.__print_info_inner(self.label, 'label')

    def normalize(self):
        self.base = MinMaxScaler().fit_transform(self.base)
        self.review = MinMaxScaler().fit_transform(self.review)
        self.amenities = MinMaxScaler().fit_transform(self.amenities)
