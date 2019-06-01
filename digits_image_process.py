import cv2
import numpy as np
import digits_ann as ANN
# inside（）函数用来确定矩形是否包含在另外一个矩形中

def inside(r1, r2):
  x1,y1,w1,h1 = r1
  x2,y2,w2,h2 = r2
  if (x1 > x2) and (y1 > y2) and (x1+w1 < x2+w2) and (y1+h1 < y2 + h2):
    return True
  else:
    return False

def wrap_digit(rect):
  x, y, w, h = rect
  padding = 5
  hcenter = x + w/2
  vcenter = y + h/2
  roi = None
  if (h > w):
    w = h
    x = hcenter - (w/2)
  else:
    h = w
    y = vcenter - (h/2)
  return (x-padding, y-padding, w+padding, h+padding)

# ann, test_data = ANN.train(ANN.create_ANN(56), 50000, 5)

ann, test_data = ANN.train(ANN.create_ANN(58), 5000, 5)



font = cv2.FONT_HERSHEY_SIMPLEX

path = "./images/numbers.png"

# path = "./images/MNISTsamples.png"
# path = "./images/numbers.jpg"

img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
bw = cv2.GaussianBlur(bw, (7,7), 0)  # 高斯模糊

# 平滑的灰度图像  使用阈值和形态学操作方法确保数字能从背景中正确区分出来
#阈值标志是一个逆二进制（inverse binary threshold）

ret, thbw = cv2.threshold(bw, 127, 255, cv2.THRESH_BINARY_INV)
thbw = cv2.erode(thbw, np.ones((2,2), np.uint8), iterations = 2)

# 进行形态学操作 需要识别分开每个图像的数字 首先要识别图像的轮廓
# 通过轮廓迭代，放弃所有完全包含在其他矩形中的矩阵
# 只添加不包含在其他矩形和不超过图像宽度的好的矩形
# 在某些测试中findcontours将整个图像作为轮廓本事，这意味着没有其他的矩形传递到inside中测试

image, cntrs, hier = cv2.findContours(thbw.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

rectangles = []

for c in cntrs:
  r = x,y,w,h = cv2.boundingRect(c)
  a = cv2.contourArea(c)
  b = (img.shape[0]-3) * (img.shape[1] - 3)
  
  is_inside = False
  for q in rectangles:
    if inside(r, q):
      is_inside = True
      break
  if not is_inside:
    if not a == b:
      rectangles.append(r)
# 现在得到一系列好的矩阵，可以对他进行迭代，并对每个已经识别的矩形定义感兴趣区域
for r in rectangles:
  x,y,w,h = wrap_digit(r)
  cv2.rectangle(img, (int(x),int(y)), (int(x+w),int( y+h)), (0, 255, 0), 2)
  roi = thbw[int(y):int(y+h), int(x):int(x+w)]

  try:
    digit_class = int(ANN.predict(ann, roi.copy())[0])
  except:
    continue

  cv2.putText(img, "%d" % digit_class, (int(x),int( y-1)), font, 1, (0, 255, 0))


cv2.imshow("thbw", thbw)
cv2.imshow("contours", img)
cv2.imwrite("sample2.jpg", img)
cv2.waitKey()
