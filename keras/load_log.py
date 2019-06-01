# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv('cnnface2.csv')

l = list(log['Epoch;acc;loss;val_acc;val_loss'])


'''
epoch = []
acc = []
loss = []
val_acc = []
val_loss = []

for i in range(0,len(l)):
    epoch.append(l[i].split(';')[0])
    acc.append(l[i].split(';')[1])
    loss.append(l[i].split(';')[2])
    val_acc.append(l[i].split(';')[3])
    val_loss.append(l[i].split(';')[4])


plt.style.use("ggplot")                          #设置绘图风格
plt.figure(figsize=(15,10))                      #设置绘图大小，单位inch
plt.plot(epoch, loss, label="train_loss")
plt.plot(epoch, val_loss, label="val_loss")
plt.plot(epoch, acc, label="train_acc")
plt.plot(epoch, val_acc, label="val_acc")
plt.title("Training Loss and Accuracy on sar classifier")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")
plt.savefig("Loss_Accuracy_mix_40-800_300e.jpg")
'''

keras.callbacks.TensorBoard(log_dir='./Graph',
                    histogram_freq= 0 ,
                    write_graph=True,
                write_images=True)
tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph',
                                         histogram_freq= 0,
                                         write_graph=True,
                                         write_images=True)

model.compile(optimizer=optim,
              loss=MultiboxLoss(NUM_CLASSES, neg_pos_ratio=2.0).compute_loss, metrics=['accuracy'])
nb_epoch = 30
history = model.fit_generator(gen.generate(True), gen.train_batches,
                              nb_epoch, verbose=1,
                             callbacks=[tbCallBack],
                             validation_data=gen.generate(False),
                              nb_val_samples=gen.val_batches,
                              nb_worker=1)
