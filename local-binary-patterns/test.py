# USAGE
# python recognize.py --training images/training --testing images/testing
# import the necessary packages
# from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
import os
import matplotlib.pyplot as plt

# import the necessary packages
from skimage import feature
import numpy as np
class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# 存储点数和半径
		self.numPoints = numPoints
		self.radius = radius

	def describe(self, image, eps=1e-7):
		# 计算本地二进制模式表示的图像，然后使用LBP表示 构建模式的直方图
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")

		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

        # cv2.imshow('lbp', lbp)
		# 标准化直方图
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# 返回局部二进制模式的直方图
		return lbp


#初始化本地二进制模式描述符数据和标签列表
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# loop over the training images
image = cv2.imread('lena.bmp')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = desc.describe(gray)
#     #


cv2.imshow('image', hist)
	# 从图像路径中提取标签，然后更新标签和数据列表
	# labels.append(imagePath.split(os.path.sep)[-2])
	# data.append(hist)
cv2.waitKey(0)   #键盘绑定函数
cv2.destroyAllWindows()




# for imagePath in paths.list_images('images/training'):
# 	# 加载图像，将其转换为灰度，并描述它
# 	image = cv2.imread(imagePath)
# 	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	# hist = desc.describe(gray)
#     #
# 	cv2.imshow('image', image)
# 	# 从图像路径中提取标签，然后更新标签和数据列表
# 	# labels.append(imagePath.split(os.path.sep)[-2])
# 	# data.append(hist)
#
# # 在数据上训练线性SVM


#
# model = LinearSVC(C=100.0, random_state=42)
#
# model.fit(data, labels)
#
#
#
# # 循环测试图像
# for imagePath in paths.list_images('images/testing'):
# 	# 加载图像，将其转换为灰度，描述，并对其进行分类
# 	# print(imagePath)
# 	image = cv2.imread(imagePath)
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	hist = desc.describe(gray)
# 	prediction = model.predict(hist.reshape(1, -1))
# 	print(prediction)
#
# 	# 显示图像和预测
# 	cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
# 		1.0, (0, 0, 255), 3)
# 	cv2.imshow("Image", image)
# 	if cv2.waitKey(0) & 0xff == ord("q"):
# 		cv2.destroyAllWindows()
#
# cv2.destroyAllWindows()



# [array([4.30500000e-02, 4.09833333e-02, 4.93333333e-03, 1.82500000e-03,
#        1.39166667e-03, 1.52500000e-03, 1.95000000e-03, 3.65833333e-03,
#        3.36666667e-03, 4.65000000e-03, 4.80000000e-03, 1.23250000e-02,
#        2.09083333e-02, 1.64083333e-02, 5.80000000e-03, 5.81666667e-03,
#        2.95000000e-03, 3.00833333e-03, 1.02500000e-03, 1.15833333e-03,
#        5.00000000e-04, 8.08333333e-04, 2.36666667e-03, 3.74750000e-02,
#        4.73333333e-02, 7.29983333e-01]), array([4.61333333e-02, 4.18583333e-02, 5.18333333e-03, 2.35833333e-03,
#        1.86666667e-03, 1.60833333e-03, 1.75000000e-03, 1.48333333e-03,
#        2.27500000e-03, 2.57500000e-03, 3.58333333e-03, 7.74166667e-03,
#        1.89166667e-02, 9.69166667e-03, 3.84166667e-03, 3.51666667e-03,
#        1.74166667e-03, 2.25833333e-03, 1.08333333e-03, 1.57500000e-03,
#        5.50000000e-04, 1.11666667e-03, 3.10000000e-03, 3.92583333e-02,
#        4.98166667e-02, 7.45116667e-01]), array([4.28333333e-02, 4.10916667e-02, 6.35000000e-03, 2.85000000e-03,
#        1.90833333e-03, 1.73333333e-03, 1.85000000e-03, 2.58333333e-03,
#        2.89166667e-03, 3.61666667e-03, 5.09166667e-03, 9.85000000e-03,
#        2.15250000e-02, 1.37416667e-02, 5.90833333e-03, 5.06666667e-03,
#        2.37500000e-03, 2.72500000e-03, 8.66666667e-04, 1.40000000e-03,
#        6.08333333e-04, 1.16666667e-03, 2.89166667e-03, 3.80750000e-02,
#        4.57333333e-02, 7.35266667e-01]), array([4.53666667e-02, 4.08166667e-02, 5.61666667e-03, 2.39166667e-03,
#        1.72500000e-03, 1.96666667e-03, 2.32500000e-03, 2.45000000e-03,
#        2.67500000e-03, 3.85000000e-03, 4.73333333e-03, 1.06500000e-02,
#        2.16333333e-02, 1.41833333e-02, 5.14166667e-03, 4.34166667e-03,
#        1.80000000e-03, 2.28333333e-03, 1.04166667e-03, 1.37500000e-03,
#        5.00000000e-04, 1.13333333e-03, 2.61666667e-03, 4.01833333e-02,
#        4.92333333e-02, 7.29966667e-01]), array([6.58583333e-02, 4.88083333e-02, 5.67500000e-03, 1.18333333e-03,
#        2.58333333e-04, 2.16666667e-04, 7.50000000e-05, 6.66666667e-05,
#        5.83333333e-05, 9.16666667e-05, 1.16666667e-04, 1.83333333e-04,
#        4.83333333e-04, 2.00833333e-03, 1.91666667e-04, 1.85000000e-03,
#        2.16666667e-04, 1.36666667e-03, 2.00000000e-04, 1.07500000e-03,
#        2.75000000e-04, 1.59166667e-03, 4.97500000e-03, 4.62666667e-02,
#        5.84750000e-02, 7.58433333e-01]), array([5.68333333e-02, 4.43666667e-02, 1.22333333e-02, 4.50000000e-03,
#        2.04166667e-03, 9.66666667e-04, 5.91666667e-04, 3.66666667e-04,
#        3.75000000e-04, 2.83333333e-04, 3.75000000e-04, 3.08333333e-04,
#        6.75000000e-04, 2.62500000e-03, 4.50000000e-04, 2.36666667e-03,
#        3.50000000e-04, 1.70000000e-03, 5.08333333e-04, 1.39166667e-03,
#        1.42500000e-03, 3.41666667e-03, 7.75000000e-03, 4.19666667e-02,
#        5.04916667e-02, 7.61641667e-01]), array([5.69500000e-02, 4.35333333e-02, 1.62416667e-02, 8.00000000e-03,
#        4.85000000e-03, 2.23333333e-03, 1.85000000e-03, 1.34166667e-03,
#        9.66666667e-04, 8.00000000e-04, 8.16666667e-04, 6.16666667e-04,
#        7.50000000e-04, 2.43333333e-03, 7.41666667e-04, 2.27500000e-03,
#        7.08333333e-04, 2.13333333e-03, 9.58333333e-04, 2.05000000e-03,
#        2.03333333e-03, 4.83333333e-03, 1.05166667e-02, 4.03500000e-02,
#        5.21000000e-02, 7.39916667e-01]), array([0.05863333, 0.03623333, 0.028525  , 0.018375  , 0.0105    ,
#        0.00549167, 0.00315   , 0.002475  , 0.00224167, 0.00195   ,
#        0.002075  , 0.00185833, 0.00206667, 0.00445833, 0.002375  ,
#        0.004275  , 0.00265833, 0.00465833, 0.00326667, 0.005     ,
#        0.006075  , 0.00913333, 0.01325   , 0.03906667, 0.04475   ,
#        0.68745833]), array([0.0269  , 0.02065 , 0.017325, 0.01695 , 0.015925, 0.013275,
#        0.01175 , 0.016125, 0.022   , 0.03045 , 0.04095 , 0.0554  ,
#        0.06555 , 0.050825, 0.032025, 0.0248  , 0.015925, 0.0151  ,
#        0.012925, 0.01515 , 0.0138  , 0.017125, 0.015075, 0.013075,
#        0.018875, 0.40205 ]), array([0.027375, 0.0209  , 0.015175, 0.015025, 0.01695 , 0.01485 ,
#        0.011725, 0.0159  , 0.022125, 0.029775, 0.0429  , 0.058   ,
#        0.065625, 0.052825, 0.0342  , 0.0253  , 0.017175, 0.014075,
#        0.0118  , 0.014125, 0.0161  , 0.01765 , 0.0147  , 0.011875,
#        0.0199  , 0.39395 ]), array([0.0256  , 0.021975, 0.016975, 0.015625, 0.0154  , 0.01465 ,
#        0.0147  , 0.016425, 0.02175 , 0.025525, 0.0349  , 0.058975,
#        0.069975, 0.05075 , 0.030825, 0.0224  , 0.015775, 0.014125,
#        0.011825, 0.013   , 0.0143  , 0.015975, 0.01435 , 0.0128  ,
#        0.019675, 0.411725]), array([0.026925, 0.02165 , 0.016175, 0.0138  , 0.015125, 0.01575 ,
#        0.014075, 0.01715 , 0.021375, 0.024325, 0.036875, 0.055775,
#        0.070875, 0.048625, 0.0318  , 0.022675, 0.01605 , 0.0134  ,
#        0.011475, 0.01335 , 0.01565 , 0.01605 , 0.014975, 0.013525,
#        0.02025 , 0.4123  ]), array([0.0252  , 0.019925, 0.0158  , 0.0149  , 0.01695 , 0.016025,
#        0.0158  , 0.016425, 0.02265 , 0.028425, 0.03525 , 0.05975 ,
#        0.07455 , 0.050925, 0.032525, 0.022225, 0.01655 , 0.01375 ,
#        0.01215 , 0.012525, 0.014075, 0.01675 , 0.01495 , 0.0121  ,
#        0.02005 , 0.399775]), array([0.0257  , 0.0212  , 0.01555 , 0.015375, 0.016375, 0.015075,
#        0.014775, 0.0162  , 0.019775, 0.024475, 0.03305 , 0.0537  ,
#        0.06825 , 0.049225, 0.028975, 0.02115 , 0.01525 , 0.013075,
#        0.011675, 0.012375, 0.013825, 0.017675, 0.014475, 0.01385 ,
#        0.019275, 0.429675]), array([0.025525, 0.021225, 0.016975, 0.0168  , 0.014575, 0.01575 ,
#        0.01385 , 0.01655 , 0.0203  , 0.0255  , 0.032625, 0.051775,
#        0.062675, 0.049675, 0.0302  , 0.021325, 0.01565 , 0.012925,
#        0.011375, 0.012775, 0.01455 , 0.01685 , 0.014375, 0.012125,
#        0.019325, 0.434725]), array([0.026275, 0.022175, 0.016075, 0.01595 , 0.0157  , 0.014275,
#        0.013175, 0.015525, 0.01815 , 0.0245  , 0.0306  , 0.05    ,
#        0.060075, 0.0474  , 0.028675, 0.022125, 0.0152  , 0.012875,
#        0.010225, 0.012   , 0.014675, 0.0146  , 0.013625, 0.014675,
#        0.020175, 0.451275]), array([0.0266  , 0.023375, 0.016125, 0.015775, 0.014875, 0.01355 ,
#        0.01275 , 0.014425, 0.0183  , 0.0229  , 0.029725, 0.045425,
#        0.054175, 0.046775, 0.02725 , 0.01985 , 0.0142  , 0.011025,
#        0.010225, 0.011975, 0.01285 , 0.014375, 0.012675, 0.0148  ,
#        0.019725, 0.476275]), array([0.024975, 0.0234  , 0.015825, 0.01415 , 0.0138  , 0.013725,
#        0.012225, 0.0144  , 0.016925, 0.02085 , 0.027425, 0.04435 ,
#        0.05315 , 0.0429  , 0.026125, 0.018275, 0.014475, 0.01105 ,
#        0.01035 , 0.01095 , 0.012275, 0.013425, 0.0126  , 0.014325,
#        0.019075, 0.498975]), array([0.0256  , 0.023625, 0.01475 , 0.01345 , 0.013425, 0.01265 ,
#        0.0109  , 0.01455 , 0.01775 , 0.021775, 0.02735 , 0.045175,
#        0.051775, 0.0438  , 0.02485 , 0.0182  , 0.013675, 0.011575,
#        0.010075, 0.00975 , 0.011875, 0.013075, 0.013175, 0.015575,
#        0.01835 , 0.50325 ]), array([0.0254  , 0.02395 , 0.015475, 0.013225, 0.0142  , 0.01415 ,
#        0.0127  , 0.0144  , 0.01835 , 0.02445 , 0.029975, 0.04525 ,
#        0.0533  , 0.044275, 0.027875, 0.019775, 0.013475, 0.0112  ,
#        0.0102  , 0.0117  , 0.01255 , 0.014025, 0.0124  , 0.01445 ,
#        0.020325, 0.482925]), array([0.026   , 0.02275 , 0.01725 , 0.01735 , 0.01465 , 0.013425,
#        0.012825, 0.014725, 0.020875, 0.02965 , 0.0414  , 0.057725,
#        0.065175, 0.0513  , 0.0326  , 0.02275 , 0.016425, 0.01525 ,
#        0.0136  , 0.015775, 0.01605 , 0.016825, 0.0152  , 0.011975,
#        0.02035 , 0.3981  ]), array([0.027175, 0.02265 , 0.015025, 0.014875, 0.015075, 0.013475,
#        0.013775, 0.014525, 0.018375, 0.0219  , 0.028925, 0.044675,
#        0.05445 , 0.044325, 0.0269  , 0.0184  , 0.013925, 0.011275,
#        0.0102  , 0.011125, 0.012575, 0.014075, 0.013375, 0.01515 ,
#        0.01915 , 0.484625]), array([0.025675, 0.0209  , 0.015475, 0.015675, 0.014175, 0.0134  ,
#        0.01395 , 0.01545 , 0.02085 , 0.028175, 0.039325, 0.0596  ,
#        0.0698  , 0.04925 , 0.03275 , 0.023875, 0.01495 , 0.014375,
#        0.012225, 0.01415 , 0.01475 , 0.015975, 0.012975, 0.012575,
#        0.021   , 0.4087  ]), array([0.02645 , 0.0208  , 0.01435 , 0.016375, 0.01465 , 0.01505 ,
#        0.013   , 0.0153  , 0.02195 , 0.0288  , 0.038225, 0.05735 ,
#        0.066775, 0.048725, 0.0331  , 0.022725, 0.01675 , 0.0157  ,
#        0.01265 , 0.014   , 0.01485 , 0.016375, 0.014   , 0.01245 ,
#        0.01985 , 0.40975 ]), array([0.02615 , 0.02045 , 0.01595 , 0.015575, 0.0131  , 0.013825,
#        0.013575, 0.015225, 0.0218  , 0.025325, 0.038   , 0.0568  ,
#        0.064225, 0.0496  , 0.0319  , 0.022675, 0.016825, 0.013425,
#        0.01165 , 0.012575, 0.015525, 0.0165  , 0.014625, 0.0123  ,
#        0.019   , 0.4234  ]), array([0.026875, 0.02155 , 0.015975, 0.0158  , 0.013775, 0.01645 ,
#        0.013975, 0.0167  , 0.021575, 0.027525, 0.034125, 0.0561  ,
#        0.067975, 0.051475, 0.032675, 0.022025, 0.017075, 0.01315 ,
#        0.012   , 0.011675, 0.01405 , 0.01695 , 0.015825, 0.01265 ,
#        0.019775, 0.412275]), array([0.02605 , 0.021475, 0.0171  , 0.01655 , 0.015425, 0.014625,
#        0.013775, 0.01725 , 0.0205  , 0.02795 , 0.038075, 0.056725,
#        0.07435 , 0.053475, 0.03265 , 0.02185 , 0.017575, 0.0129  ,
#        0.011625, 0.012275, 0.01535 , 0.01585 , 0.015825, 0.0117  ,
#        0.020725, 0.39835 ]), array([0.027525, 0.02015 , 0.017   , 0.01435 , 0.0151  , 0.014575,
#        0.0142  , 0.016525, 0.020275, 0.026375, 0.036175, 0.055575,
#        0.07435 , 0.05135 , 0.03105 , 0.02075 , 0.017125, 0.013   ,
#        0.0111  , 0.012375, 0.014525, 0.016675, 0.015325, 0.012125,
#        0.0206  , 0.411825]), array([0.025525, 0.0215  , 0.015425, 0.0147  , 0.014925, 0.015525,
#        0.01485 , 0.016825, 0.0213  , 0.02565 , 0.034125, 0.0568  ,
#        0.0711  , 0.05155 , 0.031875, 0.02165 , 0.01685 , 0.012525,
#        0.0105  , 0.01355 , 0.014725, 0.018125, 0.015225, 0.0126  ,
#        0.019575, 0.413   ]), array([0.01995833, 0.022875  , 0.01085   , 0.00951667, 0.00960833,
#        0.010775  , 0.010975  , 0.01325833, 0.01324167, 0.0174    ,
#        0.02138333, 0.03161667, 0.0685    , 0.03988333, 0.025025  ,
#        0.01611667, 0.00981667, 0.007775  , 0.00551667, 0.00578333,
#        0.0048    , 0.00566667, 0.00790833, 0.02659167, 0.02674167,
#        0.55841667]), array([0.01460833, 0.02524167, 0.00853333, 0.008     , 0.00758333,
#        0.00919167, 0.010375  , 0.01359167, 0.01278333, 0.014825  ,
#        0.0166    , 0.0284    , 0.07394167, 0.05150833, 0.02249167,
#        0.01781667, 0.01155833, 0.009825  , 0.00595   , 0.006175  ,
#        0.00445833, 0.004625  , 0.00618333, 0.02905   , 0.02644167,
#        0.56024167]), array([0.01684167, 0.022175  , 0.01205833, 0.01319167, 0.014525  ,
#        0.01251667, 0.01125   , 0.012575  , 0.014575  , 0.02065   ,
#        0.02789167, 0.044625  , 0.09095   , 0.04540833, 0.02708333,
#        0.01933333, 0.01188333, 0.009875  , 0.00790833, 0.00771667,
#        0.00759167, 0.00815833, 0.00861667, 0.02509167, 0.02490833,
#        0.4826    ]), array([0.01754167, 0.022175  , 0.010425  , 0.010475  , 0.009875  ,
#        0.00959167, 0.01135   , 0.011775  , 0.01299167, 0.01530833,
#        0.01828333, 0.02775   , 0.06135   , 0.034475  , 0.02184167,
#        0.01520833, 0.01021667, 0.00785833, 0.00543333, 0.005675  ,
#        0.005175  , 0.00604167, 0.00696667, 0.02498333, 0.02783333,
#        0.5894    ]), array([0.0357    , 0.03243333, 0.00778333, 0.00529167, 0.00529167,
#        0.00514167, 0.00474167, 0.00610833, 0.00664167, 0.007825  ,
#        0.00858333, 0.00951667, 0.02656667, 0.011775  , 0.00680833,
#        0.00609167, 0.00481667, 0.00530833, 0.0034    , 0.00353333,
#        0.00303333, 0.00394167, 0.007575  , 0.0332    , 0.037175  ,
#        0.71171667]), array([0.03774167, 0.03609167, 0.00746667, 0.00455   , 0.00354167,
#        0.00360833, 0.00390833, 0.0053    , 0.0052    , 0.00606667,
#        0.0065    , 0.008125  , 0.02188333, 0.01070833, 0.00668333,
#        0.00656667, 0.004675  , 0.00511667, 0.003125  , 0.003075  ,
#        0.002675  , 0.00349167, 0.00634167, 0.03440833, 0.04036667,
#        0.72278333]), array([0.03339167, 0.03110833, 0.00723333, 0.00585   , 0.004925  ,
#        0.0053    , 0.00595   , 0.00875833, 0.00985833, 0.01165833,
#        0.01370833, 0.01825   , 0.03583333, 0.02430833, 0.013625  ,
#        0.01124167, 0.00665   , 0.00705833, 0.00414167, 0.00433333,
#        0.00345   , 0.00500833, 0.00633333, 0.029275  , 0.03341667,
#        0.65933333]), array([0.03415833, 0.0304    , 0.00953333, 0.007275  , 0.00628333,
#        0.00673333, 0.006675  , 0.007675  , 0.00810833, 0.01050833,
#        0.01420833, 0.017675  , 0.03730833, 0.02034167, 0.01270833,
#        0.01099167, 0.00774167, 0.00741667, 0.005375  , 0.005925  ,
#        0.00484167, 0.00645   , 0.0085    , 0.03100833, 0.03805   ,
#        0.64410833])]
