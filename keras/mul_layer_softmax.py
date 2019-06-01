from keras1.models import Sequential
from keras1.layers import Dense, Dropout, Activation
from keras1.optimizers import SGD
import keras1

#简单的多层感知器 模型
# Generate dummy data
import numpy as np

x_train = np.random.random((1000, 20))
y_train = keras1.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)

x_test = np.random.random((100, 20))
y_test = keras1.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)