import os
import sys
import cv2
import numpy as np
import time as tm
from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse

'''
att_faces  size 92*112 

'''
def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX, maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)

'''

'''
def read_images(path, sz=None):
    c = 0
    X, y = [], []
    for imagePath in paths.list_images(path):
        try:
            im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            if (im is None):
                print("image " + imagePath + " is none")
            else:
                print(imagePath)
            if (sz is not None):
                im = cv2.resize(im, (200, 200))
            X.append(np.asarray(im, dtype=np.uint8))
            # y.append(imagePath.split(os.path.sep)[-2])、
            y.append(c)
        except IOError:
            print("I/O error({0}): {1}".format(errno, strerror))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        c = c + 1
    return [X,y]
def face_rec():
    names = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    '''
    X为图片的矩阵形式
    y为图片的底标
    '''
    [X,y] = read_images('image/train/att_faces/')

    y = np.asarray(y)


    model = cv2.face.EigenFaceRecognizer_create()

    model.train(np.asarray(X), np.asarray(y))

    # camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    while (True):
      # read, img = camera.read()
      img = cv2.imread('image/train/att_faces/s6/1.pgm')


      faces = face_cascade.detectMultiScale(img, 1.3, 4)
      for (x, y, w, h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = gray[x:x+w, y:y+h]
        try:

            roi = cv2.resize(roi, (92, 112), interpolation=cv2.INTER_LINEAR)
            params = model.predict(roi)
            print("Label: %s, Confidence: %.2f" % (params[0], params[1]))
            cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)

            if (params[0] == 0):
                cv2.imwrite('%s.pgm' % str(tm.time()), img)
            # cv2.imwrite('%s.pgm' % str(tm.time()), img)
        except:
            continue
      cv2.imshow("camera", img)

      if cv2.waitKey(0) & 0xFF == ord("q"):
        break

    cv2.destroyAllWindows()


def write_data(filename, data, name):
    # 打开文件
    fo = open(filename, "a+")
    # 在文件末尾写入一行
    fo.seek(0, 2)
    fo.write(name)
    fo.write(str(data))
    fo.write("\n")
    # 关闭文件
    fo.close()

def face_r():
    # 初始化本地二进制模式描述符数据和标签列表
    desc = LocalBinaryPatterns(24, 8)
    data = []
    labels = []
    # loop over the training images
    for imagePath in paths.list_images('image/train'):
        # 加载图像，将其转换为灰度，并描述它
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        #
        # 从图像路径中提取标签，然后更新标签和数据列表
        labels.append(imagePath.split(os.path.sep)[-2])
        data.append(hist)

    # write_data("face_data.csv",labels,"labels")
    # write_data("face_data.csv", data, "data")

    # 在数据上训练线性SVM
    model = LinearSVC(C=100.0, random_state=42, max_iter=10000)
    model.fit(data, labels)

    # 循环测试图像
    #
    for imagePath in paths.list_images('image/test'):
        # 加载图像，将其转换为灰度，描述，并对其进行分类
        print(imagePath)
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = desc.describe(gray)
        prediction = model.predict(hist.reshape(1, -1))
        print(prediction)
        # 显示图像和预测
        cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 3)
        cv2.imshow("Image", image)
        if cv2.waitKey(0) & 0xff == ord("q"):
            cv2.destroyAllWindows()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_rec()





