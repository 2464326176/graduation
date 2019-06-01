import cv2
import pickle
import numpy as np
import gzip

"""OpenCV ANN Handwritten digit recognition example

Wraps OpenCV's own ANN by automating the loading of data and supplying default paramters,
such as 20 hidden layers, 10000 samples and 1 training epoch.

The load data code is taken from http://neuralnetworksanddeeplearning.com/chap1.html
by Michael Nielsen
"""


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def load_data():
    with gzip.open('./data/mnist.pkl.gz') as fp:
        # 注意版本不同，需要添加传入第二个参数encoding='bytes'，否则出现编码错误
        training_data, valid_data, test_data = pickle.load(fp, encoding='bytes')
        fp.close()
    return (training_data, valid_data, test_data)


def wrap_data():
    # tr_d数组长度为50000，va_d数组长度为10000，te_d数组长度为10000
    tr_d, va_d, te_d = load_data()

    # 训练数据集
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    # 校验数据集
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    # 测试数据集
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)


def create_ANN(hidden=20):
    ann = cv2.ml.ANN_MLP_create()  # 建立模型
    ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)  # 设置训练方式为反向传播
    ann.setActivationFunction(
        cv2.ml.ANN_MLP_SIGMOID_SYM)  # 设置激活函数为SIGMOID，还有cv2.ml.ANN_MLP_IDENTITY,cv2.ml.ANNMLP_GAUSSIAN
    ann.setLayerSizes(np.array([784, hidden, 10]))  # 设置层数，输入784层，输出层10
    ann.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.1))  # 设置终止条件
    return ann


def train(ann, samples=10000, epochs=1):
    # tr:训练数据集; val:校验数据集; test:测试数据集;
    tr, val, test = wrap_data()

    for x in range(epochs):
        counter = 0
        for img in tr:
            if (counter > samples):
                break
            if (counter % 1000 == 0):
                print("Epoch %d: Trained %d/%d" % (x, counter, samples))
            counter += 1
            data, digit = img
            ann.train(np.array([data.ravel()], dtype=np.float32), cv2.ml.ROW_SAMPLE,
                      np.array([digit.ravel()], dtype=np.float32))
        print("Epoch %d complete" % x)
    return ann, test


def predict(ann, sample):
    resized = sample.copy()
    rows, cols = resized.shape
    if rows != 28 and cols != 28 and rows * cols > 0:
        resized = cv2.resize(resized, (28, 28), interpolation=cv2.INTER_CUBIC)
    return ann.predict(np.array([resized.ravel()], dtype=np.float32))