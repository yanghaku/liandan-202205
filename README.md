* 特征处理: 数值进行标准化, 类别信息进行one-hot, (description还未使用)
* 分类使用的算法:
    * 'svm'   (scikit-learn中的```LinearSVC```)
    * 'sgd'   (scikit-learn中的```SGDClassifier``)
    * 'knn'   (scikit-learn中的```KNeighborsClassifier``)
    * 'gpc'   (scikit-learn中的```GaussianProcessClassifier``)
    * 'bayes' (scikit-learn中的```MultinomialNB``)
    * 'dt'    (scikit-learn中的```DecisionTreeClassifier``)
    * 'mlp'   (scikit-learn中的```MLPClassifier``)
    * 'rf'    (scikit-learn中的```RandomForestClassifier``)
    * 'gb'    (scikit-learn中的```GradientBoostingClassifier``)
    * 'ab'    (scikit-learn中的```AdaBoostClassifier``)
    * 'dnn1'  (利用pytorch自定义的多层感知机网络)
    * 'xgb'   (利用xgboost库的梯度提升树)


当前 这些方案的f1都在0.3-0.4之间, 目前最好的是梯度提升树(才0.45......)
```shell
$$> python .\src\main.py -m gb --estimators 100
Namespace(batch=16, divide_percent=0.8, epoch=10, estimators=100, lr=0.001, method='gb',  output='****/project/src/test.res.txt', random=False, show_pic=False, task='metrics', test='****/project/src/../raw_data/test.csv', train='****/project/src/../raw_data/train.csv')
success to divide. train shape=(12050, 539), test shape=(3013, 539)
label distribution: [3660, 3799, 1428, 1789, 1068, 306]
train took 82.54704 s
train acc = 0.60249
train macro f1 = 0.54627
test took 0.02602 s
test acc = 0.56223
test macro f1 = 0.45915
label distribution: [968, 943, 335, 453, 234, 80]
predict distribution: [1009, 1312, 15, 381, 244, 52]
```

```shell
$$> python .\src\main.py -m dnn1 -e 200 --lr 0.0005
Namespace(batch=16, divide_percent=0.8, epoch=200, estimators=100, lr=0.0005, method='dnn1',  output='****/project/src/test.res.txt', random=False, show_pic=False, task='metrics', test='****/project/src/../raw_data/test.csv', train='****/project/src/../raw_data/train.csv')
USE_CUDA = True
success to divide. train shape=(12050, 539), test shape=(3013, 539)
label distribution: [3660, 3799, 1428, 1789, 1068, 306]
[epoch 1/200] average_loss=1.41492  train_acc=0.41328  test_acc=0.53236  test_f1=0.32600
[epoch 2/200] average_loss=1.16820  train_acc=0.52938  test_acc=0.54066  test_f1=0.34617
[epoch 3/200] average_loss=1.13342  train_acc=0.53776  test_acc=0.55061  test_f1=0.40374
......
......
[epoch 198/200] average_loss=0.01876  train_acc=0.99336  test_acc=0.49950  test_f1=0.43186
[epoch 199/200] average_loss=0.02189  train_acc=0.99286  test_acc=0.48556  test_f1=0.41222
[epoch 200/200] average_loss=0.02178  train_acc=0.99145  test_acc=0.49021  test_f1=0.41352
train took 741.77473 s
test took 0.23985 s
test acc = 0.49021
test macro f1 = 0.41352
label distribution: [968, 943, 335, 453, 234, 80]
predict distribution: [998, 1080, 238, 364, 273, 60]
```
