from typing import List, Tuple
import csv
import json

HEADERS = ['description', 'neighbourhood', 'latitude', 'longitude', 'type', 'accommodates', 'bathrooms', 'bedrooms',
           'amenities', 'reviews', 'review_rating', 'review_scores_A', 'review_scores_B', 'review_scores_C',
           'review_scores_D', 'instant_bookable', 'target']


class ReviewInfo:
    def __init__(self, num: int, rating: int, score_a: int, score_b: int, score_c: int, score_d: int):
        self.num = num
        self.rating = rating
        self.score_a = score_a
        self.score_b = score_b
        self.score_c = score_c
        self.score_d = score_d


class RawData:
    def __init__(self, description: str, neighbourhood: str, pos: Tuple[float, float], types: str, accommodates: int,
                 bathrooms: float, bathroom_shared: int, bedrooms: float, amenities: List[str], instant_bookable: int,
                 target: int, review_info: ReviewInfo):
        # 一段文字描述, 可能需要语义分析, (可None)
        self.description = description
        # 街区, 类别有限个, 将会one-hot编码
        self.neighbourhood = neighbourhood
        # 经纬度, 但感觉并没有什么用
        self.pos = pos
        # 类别, 有限类, 将会one-hot编码
        self.type = types
        # 容纳的个数
        self.accommodates = accommodates
        # bathroom个数
        self.bathrooms = bathrooms
        # bathroom 是否shared
        self.bathroom_shared = bathroom_shared
        # bedrooms
        self.bedrooms = bedrooms
        # 设施(词袋模型)
        self.amenities = amenities
        # 二元变量, 只能true/false
        self.instant_bookable = instant_bookable
        # target  (可None)
        self.target = target
        # 可为None
        self.review_info = review_info


def load(filename: str, has_header: bool = True) -> List[RawData]:
    ans: List[RawData] = []

    try:
        with open(filename, "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 16 and len(row) != 17:
                    raise Exception("expect columns 16 or 17, but given " + str(len(row)))

                if has_header:  # read header
                    has_header = False  # next loop will read content
                    for i in range(len(row)):
                        if row[i] != HEADERS[i]:
                            raise Exception("unexpected headers, expect " + HEADERS[i] + " but given " + row[i])
                else:
                    ans.append(parse_line(row))
    except Exception as e:
        print(e)
        exit(-1)

    return ans


def parse_line(row: List[str]) -> RawData:
    description = row[0].strip()
    if description == '':
        description = None

    neighbourhood = row[1].strip()
    assert neighbourhood != ''

    pos = (float(row[2]), float(row[3]))

    types = row[4].strip()
    assert types != ''

    accommodates = int(row[5])

    bath_num, bath_shared = parse_bathroom(row[6].strip())

    if row[7].strip() == '':
        bedrooms = 0.0
    else:
        bedrooms = float(row[7])

    amenities = json.loads(row[8].strip())

    try:
        review = ReviewInfo(int(row[9]), int(row[10]), int(row[11]), int(row[12]), int(row[13]), int(row[14]))
    except ValueError as _e:
        review = None

    if row[15].strip() == 't':
        instant_bookable = 1
    elif row[15].strip() == 'f':
        instant_bookable = 0
    else:
        raise Exception("Unexpect token in instant_bookable column: " + row[15].strip())

    if len(row) == 17:
        target = int(row[16])
    else:
        target = None

    return RawData(description, neighbourhood, pos, types, accommodates, bath_num, bath_shared, bedrooms, amenities,
                   instant_bookable, target, review)


def parse_bathroom(s: str) -> (float, int):
    if s == '':
        return 0.0, 0

    try:
        ans = float(s.strip(' ')[0])
    except ValueError as _e:
        ans = 0.5

    if "shared" in s or "Shared" in s:
        return ans, 1
    return ans, 0
