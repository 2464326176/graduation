# schroedinger_nn.py
# This reads the potential training data from genpotential.py and then sets up a neural network with 2 hidden layers.
# Additional tools to output visualize and save the network are in other files.
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import matplotlib.lines as mlines
import pandas as pd
import os

bins = 300
seedmax = 20
trainx = []
trainy = []
validx = []
validy = []
for i in range(18, 19):   # 读取 16的文件
    with open('files/test_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainx.append([float(num) for num in row])
    with open('files/test_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            trainy.append([float(num) for num in row])
    with open('files/valid_pots'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validx.append([float(num) for num in row])
    with open('files/valid_out'+str(i)+'.csv', 'r') as csvfile:
        flurg = csv.reader(csvfile)
        for row in flurg:
            validy.append([float(num) for num in row])


seed = 42

np.random.seed(seed)

tf.set_random_seed(seed)

# 有一个衰减的学习速率，使收敛速度更快，在最后的拟合更好。
# 然而，通过反复试验，简单的指数衰减不能很好地工作。
# 用指定的时间间隔进行衰减的方法。

startrate = 0.125

gs = 0

gslist = [1, 1, 2, 3, 10, 20, 40, 100, 200, 10000]

ic = 0

learnrate = tf.Variable(startrate, trainable=False)
updatelearnrate = tf.assign(learnrate, tf.multiply(learnrate, 0.75))


# set up neural network layers. There are shorter ways to do it, but this exposes the guts.

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
#1st hidden layer

W1 = tf.Variable(tf.random_uniform([bins-1, bins-1], -1./bins, 1./bins))
B1 = tf.Variable(tf.random_uniform([bins-1], -1., 1.))
L1 = tf.nn.softplus(tf.matmul(X, W1) + B1)

#2nd hidden layer
W2 = tf.Variable(tf.random_uniform([bins-1, bins-1], -1./bins, 1./bins))
B2 = tf.Variable(tf.random_uniform([bins-1], -1., 1.))
L2 = tf.nn.softplus(tf.matmul(L1, W2) + B2)

#Output layer
W3 = tf.Variable(tf.random_uniform([bins-1, bins-1], -1./bins, 1./bins))
B3 = tf.Variable(tf.random_uniform([bins-1], -1., 1.))
L3 = tf.nn.softplus(tf.matmul(L2, W3) + B3)

#Cost function
costfunc = tf.reduce_mean(tf.square(tf.subtract(L3, Y)))

optimizer = tf.train.GradientDescentOptimizer(learnrate)
trainstep = optimizer.minimize(costfunc)

#initialize
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(30000):   # 调节该参数 可以强化训练次数
    if step % 150 == 0:
        if ic == gslist[gs]:
            gs = gs + 1
            ic = 1
            sess.run(updatelearnrate)
        else:
            ic = ic + 1

    if step %100 == 0:
        print(step, 'Train loss: ', sess.run(costfunc, feed_dict={X: trainx, Y: trainy}), 'Valid loss: ', sess.run(costfunc,feed_dict={X: validx, Y: validy}))
    sess.run(trainstep, feed_dict={X: trainx, Y: trainy})

def draw():
    num = len(validx)  # 波函数矩阵的长度
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    for j in range(num):  # 可以选取特定行的数据进行拟合验证 本次选择为99行的数据  亦可以选择（88,90）区间输出多个图片便于观察分析
        plt.figure(str(j))  # 指定figure的轴
        plt.subplot(411)  # numrows, numcols, fignum
        plt.plot([validx[j][i] / max(validx[j]) for i in range(bins - 1)], 'g', )
        plt.ylabel('势能')
        plt.subplot(412)
        # 训练数据用来预测实际
        plt.plot(sess.run(L3, feed_dict={X: [validx[j]]})[0], 'r')
        plt.ylabel('预测')
        plt.subplot(413)
        # 验证实际
        plt.plot(validy[j], 'b')
        plt.ylabel('实际')
        # # 整合在一起  便于观察
        plt.subplot(414)
        netpre = pd.Series(sess.run(L3, feed_dict={X: [validx[j]]})[0])
        p1 = netpre.plot(label=r'预测',style='--', color='r')
        plt.ylabel('预测')
        grosta = pd.Series(validy[j])
        p2 = grosta.plot(secondary_y=True, style='-', color='b')
        plt.ylabel('实际')
        blue_line = mlines.Line2D([], [], linestyle='--', color='r', markersize=2, label=u'预测')
        red_line = mlines.Line2D([], [], linestyle='-', color='b', markersize=2, label=u'实际')
        plt.legend(handles=[blue_line, red_line], loc='upper left')
        plt.grid(True)  # 打开网格
        # 判断此images文件夹是否存在 不存在则创建之
        if os.path.exists('valid') == False:
            os.mkdir('valid')
        plt.savefig('valid/' + time.strftime("%Y%m%d%H%M%S_") + '_' + str(j) + '.jpg')
        # 以时间戳方式保存起来，可以查看行列 第一个数字代表行 第二个代表列
        plt.show()  # 显示图片

draw()
