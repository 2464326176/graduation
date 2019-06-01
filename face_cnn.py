import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from PIL import Image
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

np.random.seed(1337)
# class LossHistory(keras1.callbacks.Callback):
#     # 缓存train/test（loss acc）数据信息
#     def on_train_begin(self, logs={}):
#         self.losses = []
#         self.acces = []
#         self.val_losses = []
#         self.val_acces = []
#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
#         self.acces.append(logs.get('acc'))
#         self.losses.append(logs.get('val_loss'))
#         self.acces.append(logs.get('val_acc'))
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()





def cnn_Create(pool_size,kernel_size,input_shape,out_classification):
    # 构建模型
    model = Sequential()
    model.add(Conv2D(5, kernel_size, input_shape=input_shape, strides=1))  # 卷积层1
    model.add(AveragePooling2D(pool_size=pool_size, strides=2))  # 池化层1
    model.add(Dropout(0.15))

    model.add(Conv2D(9, kernel_size, strides=1))  # 卷积层2
    model.add(AveragePooling2D(pool_size=pool_size, strides=2))  # 池化层2
    model.add(Dropout(0.15))

    model.add(Flatten())  # 拉成一维数据
    model.add(Dense(out_classification))  # 全连接层2
    model.add(Dropout(0.45))
    model.add(Activation('sigmoid'))  # sigmoid评分
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def cnn_Train(model,train_data,train_labels,test_data,test_labels,epoch):
    # 训练模型
    history = LossHistory()
    model.fit(train_data,train_labels, batch_size=len(train_data), epochs=epoch, verbose=1, validation_data=(test_data,test_labels),callbacks=[history])
    return (model,history)

def cnn_Predict(model,x_test,y_test):
    # 评估模型
    score = model.evaluate(x_test,y_test, verbose=0)
    return score

