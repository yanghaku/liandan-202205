from sklearn.metrics import f1_score, accuracy_score


def get_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')


def get_acc(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def save_res(y_true, filename):
    assert y_true is not None
    assert filename is not None

    with open(filename, "w", encoding='utf-8') as f:
        for i in y_true:
            print(i, file=f)

    print("save to " + filename + " success")
