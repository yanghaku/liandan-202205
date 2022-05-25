import os.path
import torch

CACHE_DIR = '../cache'

RAW_DATA_ALL = ["../raw_data/train.csv", "../raw_data/test.csv"]

# the available methods
BASE_MODELS = [
    "svm",
    "sgd",
    "knn",
    "gpc",
    'bayes',
    'dt',
    'mlp',
    'rf',
    'gb',
    'ab',
]

METHODS = BASE_MODELS + [
    'dnn1',
    'xgb',
]

TEST_OUTPUT_FILENAME = 'test.res.txt'

USE_CUDA = torch.cuda.is_available()

# process to abs path
BASE_DIR = os.path.split(os.path.realpath(__file__))[0] + '/'
CACHE_DIR = BASE_DIR + CACHE_DIR
TEST_OUTPUT_FILENAME = CACHE_DIR + '/' + TEST_OUTPUT_FILENAME
for i in range(len(RAW_DATA_ALL)):
    RAW_DATA_ALL[i] = BASE_DIR + RAW_DATA_ALL[i]

if not os.path.isdir(CACHE_DIR):
    os.mkdir(CACHE_DIR)
