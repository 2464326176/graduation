# import the necessary packages
from skimage import feature,io,exposure
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
import cv2
class face_lbp_hist:
	'''
	P：int 圆对称相邻设定点的数量（角度空间的量化）。
    R：float 圆的半径（操作员的空间分辨率）。
    METHOD：{'default'，'ror'，'uniform'，'var'}
        *'default'：原始的局部二进制模式，它是灰度但不是旋转不变。
        *'ror'：默认实现的扩展，即灰度和旋转不变。
        *'uniform'：改进的旋转不变性，均匀的图案和角度空间的更精细量化是灰度和旋转不变。
        *'nri_uniform'：非旋转不变的均匀模式变体
	'''
	# 设置LBP参数
	def __init__(self,  radius = 3):
		# 存储点数和半径
		self.numPoints = 8 * radius
		self.radius = radius
	def describe(self, image,method,eps=1e-7):
		# 计算本地二进制模式表示的图像，然后使用LBP表示 构建模式的直方图
		lbp = feature.local_binary_pattern(exposure.equalize_hist(image), self.numPoints,self.radius, method)
		# selem = disk(3)
		# img_eq = rank.equalize(lbp, selem=selem)
		#直方图均衡化
		# img_rescale = exposure.equalize_hist(lbp)
		return lbp


p1 = face_lbp_hist()
img = io.imread('yh.pgm')
plt.figure('1')
plt.imshow(p1.describe(img,'default'))
plt.show()
#
# plt.figure('1')
# plt.subplot(251)
# plt.title("ori image")
# plt.imshow(image)
#
# plt.subplot(252)
# plt.title('default lbp')
# plt.imshow(p1.describe(img,'default')[0])
#
# plt.subplot(253)
# plt.title("ror lbp")
# plt.imshow(p1.describe(img,'ror')[0])
#
# plt.subplot(254)
# plt.title("uniform lbp")
# plt.imshow(p1.describe(img,'uniform')[0])
#
#
# plt.subplot(255)
# plt.title("nri_uniform lbp")
# plt.imshow(p1.describe(img,'nri_uniform')[0])
#
# plt.subplot(256)
# plt.title("gray")
# plt.imshow(img)
#
#
# plt.subplot(257)
# plt.title('default eqe')
# plt.imshow(p1.describe(img,'default')[1])
#
# plt.subplot(258)
# plt.title("ror eqe")
# plt.imshow(p1.describe(img,'ror')[1])
#
# plt.subplot(259)
# plt.title("uniform eqe")
# plt.imshow(p1.describe(img,'uniform')[1])
#
#
# plt.subplot(2,5,10)
# plt.title("nri_uniform eqe")
# plt.imshow(p1.describe(img,'nri_uniform')[1])
#
# plt.show()