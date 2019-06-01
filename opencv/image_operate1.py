import numpy as np
import cv2
# 1，0，-1
# cv2.IMREAD_COLOR : 默认使用该种标识。加载一张彩色图片，忽视它的透明度。
# cv2.IMREAD_GRAYSCALE : 加载一张灰度图。
# cv2.IMREAD_UNCHANGED : 加载图像，包括它的Alpha通道。    
img = cv2.imread('../timg2.jpg', 2)



cv2.imshow('image', img)
cv2.waitKey(0)   #键盘绑定函数
cv2.destroyAllWindows()








