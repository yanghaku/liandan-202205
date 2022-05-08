from typing import List
import numpy as np
from . import raw_data, encoder


class MatrixData:
    def __init__(self, filename, has_header=True):
        raw = raw_data.load(filename, has_header)

        # 包含的列: accommodates,bathrooms,bathroom_shared,bedrooms,instant_bookable
        # 经纬度暂不加入('latitude', 'longitude')
        self.base = self._get_base_matrix(raw)

        # 包含的列 'reviews', 'review_rating', 'review_scores_A', 'review_scores_B', 'review_scores_C', 'review_scores_D'
        self.review = self._get_review_matrix(raw)

        # onehot之后的矩阵
        self.types = self._get_types_matrix(raw)
        # onehot之后的矩阵
        self.neighbourhood = self._get_neighbourhood_matrix(raw)
        # onehot之后的矩阵
        self.amenities = self._get_amenities_matrix(raw)
        # 语义的处理
        self.description = self._get_description_matrix(raw)

        # label (测试集可以是none)
        if raw[0].target is None:
            self.label = None
        else:
            self.label = np.zeros((len(raw, ))).astype(np.longlong)
            for i in range(len(raw)):
                self.label[i] = raw[i].target

    @staticmethod
    def _get_base_matrix(raw: List[raw_data.RawData]) -> np.ndarray:
        ans = np.zeros((len(raw), 5)).astype(np.float64)
        for i in range(len(raw)):
            ans[i][0] = float(raw[i].accommodates)
            ans[i][1] = raw[i].bathrooms
            ans[i][2] = raw[i].bathroom_shared
            ans[i][3] = raw[i].bedrooms
            ans[i][4] = raw[i].instant_bookable
        return ans

    def _get_review_matrix(self, raw: List[raw_data.RawData]) -> np.ndarray:
        pass

    @staticmethod
    def _get_types_matrix(raw: List[raw_data.RawData]) -> np.ndarray:
        lst = []
        for i in raw:
            lst.append([i.type])
        return encoder.TypesEncoder().transform(lst)

    @staticmethod
    def _get_neighbourhood_matrix(raw: List[raw_data.RawData]) -> np.ndarray:
        lst = []
        for i in raw:
            lst.append([i.neighbourhood])
        return encoder.NeighbourhoodEncoder().transform(lst)

    def _get_amenities_matrix(self, raw: List[raw_data.RawData]) -> np.ndarray:
        pass

    def _get_description_matrix(self, raw: List[raw_data.RawData]) -> np.ndarray:
        # todo
        pass
