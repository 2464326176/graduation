from skimage import data,exposure
import matplotlib.pyplot as plt
img=data.moon()
plt.figure("hist",figsize=(8,8))


arr=img.flatten()
plt.subplot(221)
plt.imshow(img,plt.cm.gray)  #原始图像
plt.subplot(222)
plt.hist(arr, bins=256, normed=1,edgecolor='None',facecolor='red') #原始图像直方图

img1=exposure.equalize_hist(img)
arr1=img1.flatten()
plt.subplot(223)
plt.imshow(img1,plt.cm.gray)  #均衡化图像
plt.subplot(224)
plt.hist(arr1, bins=256, normed=1,edgecolor='None',facecolor='red') #均衡化直方图

plt.show()
