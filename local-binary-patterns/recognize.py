# USAGE
# python recognize.py --training images/training --testing images/testing
# import the necessary packages
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os



# 构造参数解析并解析参数
ap = argparse.ArgumentParser()

ap.add_argument("-t", "--training", required=True,
	help="path to the training images")
ap.add_argument("-e", "--testing", required=True,
	help="path to the tesitng images")
args = vars(ap.parse_args())

#初始化本地二进制模式描述符数据和标签列表
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# loop over the training images
for imagePath in paths.list_images(args["training"]):
	# 加载图像，将其转换为灰度，并描述它
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	# 从图像路径中提取标签，然后更新标签和数据列表
	labels.append(imagePath.split(os.path.sep)[-2])
	data.append(hist)

# # 在数据上训练线性SVM
# model = LinearSVC(C=100.0, random_state=42)
# model.fit(data, labels)

# # 循环测试图像
# for imagePath in paths.list_images(args["testing"]):
# 	# 加载图像，将其转换为灰度，描述，并对其进行分类
# 	image = cv2.imread(imagePath)
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	hist = desc.describe(gray)
# 	prediction = model.predict(hist.reshape(1, -1))
#
# 	# 显示图像和预测
# 	cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
# 		1.0, (0, 0, 255), 3)
# 	cv2.imshow("Image", image)
# 	cv2.waitKey(0)