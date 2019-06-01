import cv2
import numpy as np
import os

def LBP(image):

    [W, H] = image.shape    #获得图像长宽

    xx = [-1,  0,  1, 1, 1, 0, -1, -1]
    yy = [-1, -1, -1, 0, 1, 1,  1,  0]    #xx, yy 主要作用对应顺时针旋转时,相对中点的相对值.
    res = np.zeros((W - 2, H - 2),dtype="uint8")  #创建0数组,显而易见维度原始图像的长宽分别减去2，并且类型一定的是uint8,无符号8位,opencv图片的存储格式.
    for i in range(1, W - 2):
        for j in range(1, H - 2):
            temp = ""
            for m in range(8):
                Xtemp = xx[m] + i
                Ytemp = yy[m] + j    #分别获得对应坐标点
                if image[Xtemp, Ytemp] > image[i, j]: #像素比较
                    temp = temp + '1'
                else:
                    temp = temp + '0'
            #print int(temp, 2)
            res[i - 1][j - 1] =int(temp, 2)   #写入结果中
    return res


img = cv2.imread('typical_image/lena.bmp')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
res = LBP(img.copy())
cv2.imshow("lbp", res)
cv2.waitKey(0)
cv2.destroyAllWindows()

