# import the necessary packages
from skimage import feature,io,exposure
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
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
	def __init__(self,  radius = 3, method = 'ror'):
		# 存储点数和半径
		self.numPoints = 8 * radius
		self.radius = radius
		self.method = method
	def describe(self, image, eps=1e-7):
		# 计算本地二进制模式表示的图像，然后使用LBP表示 构建模式的直方图
		lbp = feature.local_binary_pattern(image, self.numPoints,self.radius, self.method)
		# selem = disk(3)
		# img_eq = rank.equalize(lbp, selem=selem)
		#直方图均衡化
		img_rescale = exposure.equalize_hist(lbp)
		# (hist, _) = np.histogram(img_rescale.ravel(),bins=256)
		# # 直方图标准化（归一化）
		# # return hist
		# min = hist.min()
		# max = hist.max()
		# hist_normalization = [(da - min) / (max - min) for da in hist]
		# # (hist, _) = np.histogram(img_rescale.ravel(),
		# # 						 bins=np.arange(0, self.numPoints + 3),
		# # 						 range=(0, self.numPoints + 2))
		# # hist /= (hist.sum() + eps)

		return img_rescale


	def plot_img_and_hist(image, axes, bins=256):
		ax_img, ax_hist = axes
		ax_cdf = ax_hist.twinx()
		# Display image
		ax_img.imshow(image, cmap=plt.cm.gray)
		ax_img.set_axis_off()
		# Display histogram
		ax_hist.hist(image.ravel(), bins=bins)
		ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
		ax_hist.set_xlabel('Pixel intensity')
		# xmin, xmax = dtype_range[image.dtype.type]
		# ax_hist.set_xlim(xmin, xmax)
		# Display cumulative distribution
		img_cdf, bins = exposure.cumulative_distribution(image, bins)
		ax_cdf.plot(bins, img_cdf, 'r')
		return ax_img, ax_hist, ax_cdf

'''
lbp提取并画出均衡化后的直方图
'''
# img = io.imread('lena.bmp')
# lbp = feature.local_binary_pattern(img, 24,3, 'uniform')
# img_rescale = exposure.equalize_hist(lbp)
# # img_rescale.hist(image.ravel(), bins=bins)
# # Display results
# fig = plt.figure(figsize=(4,4))
# axes = np.zeros((2, 1), dtype=np.object)
# axes[0, 0] = plt.subplot(2,1,1)
# axes[1, 0] = plt.subplot(2,1,2)
# ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 0])
# ax_img.set_title('global equalize')
# ax_hist.set_ylabel('Number of pixels')
# # prevent overlap of y-axis labels
# fig.tight_layout()
# plt.show()

#
# p1 = face_lbp_hist()
#
# img = io.imread('att_faces/s1/1.pgm')
# p2 = p1.describe(img)

