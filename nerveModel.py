import cv2
import pickle
import numpy as np
import gzip
# from localbinarypatterns import LocalBinaryPatterns
# 创建10个元素的零数组,在期望的位置设置1 ,这个数组会被用作输出层的类标签
def vectorized_result(j):
    e = np.zeros((40, 1))
    e[j] = 1.0
    return e
# def __init__(self, desc):
#     self.desc = LocalBinaryPatterns(24, 8) # 初始化局部二进制模式描述符数据和标签列表
def wrap_data(data, labels):
    data, labels = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)
'''
加入神经网络  识别人脸  
输入层  神经元 10304
输出层  神经元 40
隐藏层  神经元 80 
训练采用反向传播方式  backprop和rprop 都是反向传播算法 
都会根据分类的误差来改变权重（有监督的学习方式）
train（samples，layout, responses）
首先 和SVM一样  ANN是opencv的 StatModel类继承的函数  train和predict
其次  只用了samples 参与训练的是无监督学习 提供了layout和response就是有监督的学习
由于使用了ANN，因此可以设置反向传播算法的类型（BACKPROP或RPROP）,这两种方式只有在有监督的学习中才可以使用
反向传播的两个阶段 （1）计算预测误差，并在输入层和输出层两个方向更新网络
                  （2）更新相应的神经元的权重
'''
# 360张图片  26个节点   40个输出结果
# 创建ann神经网络模型 设置输入层、隐藏层、输出层神经元
def create_ANN(intput_neurons,hidden_neurons,output_neurons):
    ann = cv2.ml.ANN_MLP_create()
    # ann.setLayerSizes(np.array([10304, hidden, 40]))
    ann.setLayerSizes(np.array([intput_neurons, hidden_neurons, output_neurons]))
    ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP, 0.001)
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    ann.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1, 0.000001))
    ann.setBackpropMomentumScale(1.0)
    ann.setBackpropWeightScale(0.00001)
    return ann

# 训练数据
def train(ann, data, epochs,):
    for x in range(0,epochs):
        for img_info in data:
            s,r = img_info
            ann.train(s, cv2.ml.ROW_SAMPLE, r)
            # ann.train(np.array([np.array(data).ravel()],dtype=np.float32),np.array(sign))
        print("Epoch %d complete" % x)
    return ann

def test(ann, test_data):

    sample = np.array(test_data[0][0].ravel(), dtype=np.float32).reshape(28, 28)
    cv2.imshow("sample", sample)
    cv2.waitKey()
    print(ann.predict(np.array([test_data[0][0].ravel()], dtype=np.float32)))

def predict(ann, sample):

    # resized = sample.copy()
    # rows, cols = resized.shape
    # if (rows != 112 or cols != 92) and rows * cols > 0:
    #     resized = cv2.resize(resized, (112, 92), interpolation=cv2.INTER_LINEAR)
    return ann.predict(sample)



