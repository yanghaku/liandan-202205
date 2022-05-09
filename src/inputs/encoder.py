from typing import List, Union

import numpy
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
import config
from . import raw_data

NEIGHBOURHOOD_ENCODER_CACHE_FILENAME = config.CACHE_DIR + '/neighbourhood.set'
TYPE_ENCODER_CACHE_FILENAME = config.CACHE_DIR + '/types.set'
AMENITIES_ENCODER_CACHE_FILENAME = config.CACHE_DIR + '/amenities.set'


# 从缓存文件中读取
def _load_from_file(filename: str) -> Union[List[str], None]:
    ans = []
    try:
        with open(filename, "r", encoding='utf-8') as f:
            for i in f:
                k = i.strip()
                if k != '':
                    ans.append(k)
    except IOError as _e:
        return None
    return ans


# 保存到缓存中
def _save_to_file(st: List[str], filename: str):
    try:
        with open(filename, "w", encoding='utf-8') as f:
            for i in st:
                print(i, file=f)
        print("cache to file: " + filename)
    except IOError as e:
        print(e)
        exit(-1)


# 从所有的数据集里面获取所有的类别数据
def _get_all_attr(key: str) -> List[str]:
    ans = set()
    for d in config.RAW_DATA_ALL:
        if key == 'type':
            for k in raw_data.load(d):
                ans.add(k.type)
        elif key == 'neighbourhood':
            for k in raw_data.load(d):
                ans.add(k.neighbourhood)
    return list(ans)


class NeighbourhoodEncoder:
    def __init__(self, with_cache=True):
        # 如果用cache就尝试读取cache里面
        if with_cache:
            st = _load_from_file(NEIGHBOURHOOD_ENCODER_CACHE_FILENAME)
            if st is None:  # 如果解析失败, 就重新加载
                st = _get_all_attr('neighbourhood')
                # 并且存储到缓存文件中
                _save_to_file(st, NEIGHBOURHOOD_ENCODER_CACHE_FILENAME)
        else:
            st = _get_all_attr('neighbourhood')

        X = [[x] for x in st]
        self._encoder: OneHotEncoder = OneHotEncoder().fit(X)

    def transform(self, X) -> numpy.ndarray:
        return self._encoder.transform(X).toarray()


class TypesEncoder:
    def __init__(self, with_cache=True):
        if with_cache:
            st = _load_from_file(TYPE_ENCODER_CACHE_FILENAME)
            if st is None:
                st = _get_all_attr('type')
                _save_to_file(st, TYPE_ENCODER_CACHE_FILENAME)
        else:
            st = _get_all_attr('type')

        X = [[x] for x in st]
        self._encoder: OneHotEncoder = OneHotEncoder().fit(X)

    def transform(self, X) -> numpy.ndarray:
        return self._encoder.transform(X).toarray()


class AmenitiesEncoder:
    def __init__(self, with_cache=True):
        if with_cache:
            st = _load_from_file(AMENITIES_ENCODER_CACHE_FILENAME)
            if st is None:
                st = self._get_sets()
                _save_to_file(st, AMENITIES_ENCODER_CACHE_FILENAME)
        else:
            st = self._get_sets()

        self._encoder: CountVectorizer = CountVectorizer(vocabulary=st)
        self._encoder.fit(st)

    @staticmethod
    def _get_sets() -> List[str]:
        ans = set()
        for d in config.RAW_DATA_ALL:
            for k in raw_data.load(d):
                for a in k.amenities:
                    ans.add(a)
        return list(ans)

    def transform(self, X: List[List[str]]) -> numpy.ndarray:
        docs = []
        for i in X:
            docs.append('\n'.join(i))
        return self._encoder.transform(docs).toarray()
