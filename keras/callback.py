import numpy as np
from keras1.models import Sequential
from keras1.layers import Dense, Activation, Flatten
from keras1.layers import Conv2D, MaxPooling2D,AveragePooling2D
from PIL import Image
import keras1
from keras1.layers import Dense, Dropout, Flatten
from keras1.layers import Conv2D, MaxPooling2D
from keras1.optimizers import SGD
import keras1
class LossHistory(keras1.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acces = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acces.append(logs.get('acc'))
model = Sequential()
model.add(Dense(10, input_dim=784, init='uniform'))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

history = LossHistory()
model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=0, callbacks=[history])

# print(history.losses)
# outputs
