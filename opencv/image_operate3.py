import numpy as np
import cv2

count = 0
img = cv2.imread('../timg2.jpg', 2)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)

k = cv2.waitKey(0) & 0xFF
path = 'ly/%s.pgm' % str(count)

if k == 27:
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite(path , img)
