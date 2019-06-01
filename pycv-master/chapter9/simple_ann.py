import cv2
import numpy as np

'''

训练采用反向传播方式  backprop和pprop 都是反向传播算法 
都会根据分类的误差来改变权重（有监督的学习方式）
train（samples，layout, responses）
首先 和SVM一样  ANN是opencv的 StatModel类继承的函数  train和predict
其次  只用了samples 参与训练的是无监督学习 提供了layout和response就是有监督的学习
由于使用了ANN，因此可以设置反向传播算法的类型（BACKPROP或RPROP）,这两种方式只有在有监督的学习中才可以使用
反向传播的两个阶段 （1）计算预测误差，并在输入层和输出层两个方向更新网络
                  （2）更新相应的神经元的权重
'''



ann = cv2.ml.ANN_MLP_create()

# setLayerSizes 函数 通过Numpy  数组定义各层大小
# 数组的第一个元素为元素输入层的大小
# 最后一个元素设置输出层的大小
# 中间元素定义隐藏层的大小

cv2.ml.ANN_MLP_create().setLayerSizes(np.array([9, 5, 9], dtype=np.uint8))

ann.setLayerSizes(np.array([9, 5, 9], dtype=np.uint8))


ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)

ann.train(np.array([[1.2, 1.3, 1.9, 2.2, 2.3, 2.9, 3.0, 3.2, 3.3]], dtype=np.float32),
  cv2.ml.ROW_SAMPLE,
  np.array([[0,0,0,0,0,1,0,0,0]], dtype=np.float32))

print(ann.predict(np.array([[1.4, 1.5, 1.2, 2., 2.5, 2.8, 3., 3.1, 3.8]], dtype=np.float32)))

