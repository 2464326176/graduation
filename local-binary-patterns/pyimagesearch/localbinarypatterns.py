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

		# 标准化直方图
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# 返回局部二进制模式的直方图
		return hist