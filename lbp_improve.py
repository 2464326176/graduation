import os
import cv2
import numpy as np
import time as tm
from imutils import paths
import nerveModel
from face_lbp_hist import face_lbp_hist
import math
def write_data(filename, data, name):
    # 打开文件
    fo = open(filename, "a+")
    # 在文件末尾写入一行
    fo.seek(0, 2)
    fo.write(name)
    fo.write(str(data))
    fo.write("\n")
    # 关闭文件
    fo.close()

# 创建40个元素的零数组,在期望的位置设置1 ,这个数组会被用作输出层的类标签
def vectorized_result(j):
    e = np.zeros((1, 40))
    e[0,j] = 1.0
    return e



#初始化本地二进制模式描述符数据和标签列表
desc = face_lbp_hist()
data = []
labels = []

# 加载待训练的图片信息
for imagePath in paths.list_images('att_faces'):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	# labels.append(imagePath.split(os.path.sep)[-2])
	classification = vectorized_result(int(imagePath.split(os.path.sep)[-2].split('s')[1])-1)
	# data.append((np.array([np.array(hist).ravel()], dtype=np.float32), np.array(classification, dtype=np.float32)))
	data.append((np.array([hist], dtype=np.float32), np.array(classification, dtype=np.float32)))

def ceshi():
	# 训练数据
	intput_neurons = len(data[0][0][0])
	output_neurons = len(data[0][1][0])
	hidden_neurons = int(np.sqrt(intput_neurons * output_neurons))
	for i in range(hidden_neurons,170):
		Identification = 0

		ann = nerveModel.train(nerveModel.create_ANN(intput_neurons,i,output_neurons), data,5)
		for imagePath in paths.list_images('att_face_image/test'):
			image = cv2.imread(imagePath)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			hist = desc.describe(gray)
			# classification = vectorized_result(int(imagePath.split(os.path.sep)[-2].split('s')[1]) - 1)
			classnum = int(imagePath.split(os.path.sep)[-1].split('.')[0].split('s')[1]) - 1
			# data.append(, np.array(classification, dtype=np.float32)))

			# 预测
			# prediction = nerveModel.predict(ann, (np.array([hist], dtype=np.float32)))
			prediction = ann.predict(np.array([hist], dtype=np.float32))
			# prediction = ann.predict(np.array([np.array(hist).ravel()], dtype=np.float32))
			print(prediction[0])
		# 	if int(prediction[0]) == classnum:
		# 		Identification += 1
		# print('本次实验，识别率为： {:.2%}'.format(Identification / 40))
		cv2.waitKey(0)
		cv2.destroyAllWindows()

if __name__ == "__main__":
	ceshi()