import os,cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from face_lbp_hist import face_lbp_hist
import face_cnn
# 创建40个元素的零数组,在期望的位置设置1 ,这个数组会被用作输出层的类标签
def vectorized_result(j):
	v = np.zeros((40, 1))
	v[j] = 1.0
	return v
#加载数据,收集信息
def load_data():
	# 初始化局部二值模式数据和标签列表
	desc = face_lbp_hist()
	train_data = []
	train_labels = []
	test_data = []
	test_labels = []
	# 训练的图片信息
	for imagePath in paths.list_images('att_face_image/train'):
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		lbp = desc.describe(gray)
		# labels.append(imagePath.split(os.path.sep)[-2])
		train_classification = vectorized_result(int(imagePath.split(os.path.sep)[-2].split('s')[1])-1)
		# data.append((np.array([np.array(hist).ravel()], dtype=np.float32), np.array(classification, dtype=np.float32)))
		train_labels.append(list(train_classification.ravel()))
		train_data.append(lbp)
	for imagePath in paths.list_images('att_face_image/test'):
		image = cv2.imread(imagePath)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		lbp = desc.describe(gray)
		test_classification = vectorized_result(int(imagePath.split(os.path.sep)[-1].split('.')[0].split('s')[1]) - 1)
		test_labels.append(list(test_classification.ravel()))
		test_data.append(lbp)
	return (np.array(train_data),np.array(train_labels)),(np.array(test_data),np.array(test_labels))


(train_data,train_labels),(test_data,test_labels)= load_data()
train_data=train_data[:,:,:,np.newaxis]
test_data=test_data[:,:,:,np.newaxis] # 4个维度，个数、宽度、高度、通道数
#创建卷积神经网络
img_high, img_width = len(train_data[0]),len(train_data[0][0])
pool_size = (2, 2)  # 池化层的大小
kernel_size = (5, 5)  # 卷积核的大小
input_shape = (img_high, img_width,1)  # 图片的高、宽、通道数
out_classification = 40 # 输出分类的数目
epochs = 500; # 迭代次数
model_Create = face_cnn.cnn_Create(pool_size,kernel_size,input_shape,out_classification)

#卷积神经网络的训练
model = face_cnn.cnn_Train(model_Create,train_data,train_labels,test_data,test_labels,epochs)

#卷积神经网络的预测
model_Predict = face_cnn.cnn_Predict(model[0],test_data,test_labels)

print('Test score:', model_Predict[0],'Test accuracy:', model_Predict[1])

#绘制曲线
model[1].loss_plot('epoch')





