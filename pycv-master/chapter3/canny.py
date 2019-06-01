import cv2
import numpy as np

# canny 边缘检测
# 1.高斯滤波器对图像去噪
# 2.计算梯度
# 3.在边缘上使用非最大抑制（NMS）
# 4.在检测到的边缘上使用双阈值去除假阳性（false positive）
# 5.最后分析所有的边缘及其之间的连接，确保真正的边缘并消除不明显的边缘
img = cv2.imread("../images/statue_small.jpg", 0)
cv2.imwrite("canny.jpg", cv2.Canny(img, 200, 300))
cv2.imshow("canny", cv2.imread("canny.jpg"))
cv2.waitKey()
cv2.destroyAllWindows()
