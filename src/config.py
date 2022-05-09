import os.path

CACHE_DIR = '../cache'

RAW_DATA_ALL = ["../raw_data/train.csv", "../raw_data/test.csv"]

# the available methods
METHODS = [
    "svm",
    "sgd",
    "knn",
    "gpc",
    'bayes',
    'dt',
    'mlp',
]

TEST_OUTPUT_FILENAME = 'test.res.txt'

# process to abs path
BASE_DIR = os.path.split(os.path.realpath(__file__))[0] + '/'
CACHE_DIR = BASE_DIR + CACHE_DIR
TEST_OUTPUT_FILENAME = BASE_DIR + TEST_OUTPUT_FILENAME
for i in range(len(RAW_DATA_ALL)):
    RAW_DATA_ALL[i] = BASE_DIR + RAW_DATA_ALL[i]

if not os.path.isdir(CACHE_DIR):
    os.mkdir(CACHE_DIR)
