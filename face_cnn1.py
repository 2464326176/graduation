import numpy as np
from keras1.models import Sequential
from keras1.layers import Dense, Activation, Flatten
from keras1.layers import Conv2D, MaxPooling2D,AveragePooling2D
from PIL import Image
import keras1
from keras1.layers import Dense, Dropout, Flatten
from keras1.layers import Conv2D, MaxPooling2D
from keras1.optimizers import SGD
class LossHistory(keras1.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acces = []
        self.val_losses = []
        self.val_acces = []
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acces.append(logs.get('acc'))
        self.losses.append(logs.get('val_loss'))
        self.acces.append(logs.get('val_acc'))
#随机种子
np.random.seed(1337)  # for reproducibility

batch_size = 360 # 批处理样本数量
nb_classes = 40  # 分类数目

epochs = 700# 迭代次数
img_rows, img_cols = 112,92  # 输入图片样本的宽高
# img_rows, img_cols = 256,1  # 输入图片样本的宽高

nb_filters = 32  # 卷积核的个数
pool_size = (2, 2)  # 池化层的大小
kernel_size = (5, 5)  # 卷积核的大小
input_shape = (img_rows,img_cols,1)  # 输入图片的维度



def cnn_Create():
    # 构建模型

    model = Sequential()
    model.add(Conv2D(8, kernel_size, input_shape=input_shape, strides=1))  # 卷积层1
    model.add(AveragePooling2D(pool_size=pool_size, strides=2))  # 池化层

    model.add(Conv2D(16, kernel_size, strides=1))  # 卷积层2
    model.add(AveragePooling2D(pool_size=pool_size, strides=2))  # 池化层

    model.add(Flatten())  # 拉成一维数据
    model.add(Dense(nb_classes))  # 全连接层2
    model.add(Activation('sigmoid'))  # sigmoid评分

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # model.add(Conv2D(12, kernel_size, input_shape=input_shape, strides=1))  # 卷积层1
    # model.add(AveragePooling2D(pool_size=pool_size, strides=2))  # 池化层
    # model.add(Dropout(0.25))
    #
    # model.add(Conv2D(24, kernel_size, strides=1))  # 卷积层2
    # model.add(AveragePooling2D(pool_size=pool_size, strides=2))  # 池化层
    # model.add(Dropout(0.25))
    #
    # model.add(Flatten())  # 拉成一维数据
    # model.add(Dropout(0.5))
    # model.add(Dense(nb_classes))  # 全连接层2
    # model.add(Activation('sigmoid'))
    # # 编译模型
    # model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

def cnn_Train(model,train_data,train_labels,test_data,test_labels):
    # 训练模型
    history = LossHistory()
    model.fit(train_data,train_labels, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_data,test_labels),callbacks=[history])
    return (model,history)

def cnn_Predict(model,x_test,y_test):
    # 评估模型
    score = model.evaluate(x_test,y_test, verbose=0)
    return score


# test_pred = test_pred.argmax(axis=1)   # 获取概率最大的分类，获取每行最大值所在的列
# iu = 0
# for i in range(len(test_pred)):
#     # oneimg = X_test[i,:,:,0]*256
#     # im = Image.fromarray(oneimg)
#     # im.show()
# 	if i = test_pred[i]:
# 		iu += 1
#     # print('第%d个人识别为第%d个人' % (i, test_pred[i]))