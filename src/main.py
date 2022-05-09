import argparse
import config
from inputs.dataset import Dataset
from classifier.classifier_factory import create
from outputs.helper import save_res

task_choices = ["metrics", "train", "test"]


def task_choice_helper():
    return "1.metrics: divide train dataset to train/test, score the method; " + \
           "2.train: train all train dataset and save model to file; " + \
           "3.test: load the pre-trained model and test all test dataset and save result to file;"


def run(opt):
    print(opt)

    if opt.task == task_choices[0]:
        classifier = create(opt.method, load_pretrained=False)
        dataset = Dataset(filename=opt.train)
        train, test = dataset.divide_to_train_test(opt.divide_percent, opt.random)
        classifier.train(train)
        classifier.test(test)

    elif opt.task == task_choices[1]:
        classifier = create(opt.method, load_pretrained=False)
        dataset = Dataset(filename=opt.train)
        classifier.train(dataset, save_model=True)

    elif opt.task == task_choices[2]:
        classifier = create(opt.method, load_pretrained=True)
        dataset = Dataset(filename=opt.test)
        y_pred = classifier.test(dataset)
        save_res(y_pred, opt.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--task", choices=task_choices, default=task_choices[0], help=task_choice_helper())
    parser.add_argument("-m", "--method", help="classify method to use", required=True, choices=config.METHODS)

    parser.add_argument("--train", default=config.RAW_DATA_ALL[0],
                        help="if train, specify the train data file, default is " + config.RAW_DATA_ALL[0])
    parser.add_argument("--test", default=config.RAW_DATA_ALL[1],
                        help="if test, specify the test data file, default is " + config.RAW_DATA_ALL[1])
    parser.add_argument("-o", "--output", default=config.TEST_OUTPUT_FILENAME,
                        help="output file for test, default is " + config.TEST_OUTPUT_FILENAME)

    parser.add_argument("--divide-percent", type=float, default=0.8,
                        help="divide the dataset train/all percent, default is 0.8")
    parser.add_argument("--random", type=bool, default=False, help="random divide the dataset, default is False")

    run(parser.parse_args())
