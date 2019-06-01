import numpy as np
import cv2
from matplotlib import pyplot as plt


img = cv2.imread('../timg2.jpg', 0)

plt.imshow(img, cmap='gray', interpolation='bicubic')

plt.xticks([]), plt.yticks([])  #hide tick values on x and y axis
plt.show()