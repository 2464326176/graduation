import os,shutil
import random
import cv2
class image_normalization:
    # step 1:采集图像 并规格化
    def generate(self,imgpath):
        face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
        # eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
        img = cv2.imread(imgpath)
        if img.ndim != 2:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            return f

    def copyFile(self,srcfile, dstfile):
        if not os.path.exists('%s/test' % dstfile):
            os.makedirs('%s/test' % dstfile)
        if not os.path.exists('%s/train' % dstfile):
            os.makedirs('%s/train' % dstfile)
        for root, dirs, files in os.walk(srcfile, topdown=False):
            for name in dirs:
                num = random.randint(1, 10)
                testfilepath = os.path.join(root, name, '%s.pgm' % str(num))
                # print('%s/test/%s.pgm' % (dstfile,name))
                cv2.imwrite('%s/test/%s.pgm' % (dstfile,name), self.generate(testfilepath))
                # shutil.copy(testfilepath, '%s/test/%s.pgm' % (dstfile,name))
                if not os.path.exists('%s/train/%s' % (dstfile,name)):
                    os.makedirs('%s/train/%s' % (dstfile,name))
                for i in range(1,11):
                    if i != num:
                        trainfilepath = os.path.join(root, name, '%s.pgm' % str(i))
                        cv2.imwrite('%s/train/%s/%s.pgm' % (dstfile,name,i), self.generate(trainfilepath))
                        # shutil.copy(trainfilepath, '%s/train/%s' % (dstfile,name))
                print("copy %s -> %s"%( os.path.join(root, name),dstfile))
        print('copy complete')
        # 文件复制 移动
    def moveFile(self,srcfile, dstfile):
        if not os.path.isfile(srcfile):
            print("%s not exist!" % (srcfile))
        else:
            fpath, fname = os.path.split(dstfile)
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            shutil.move(srcfile, dstfile)
            print("move %s -> %s" % (srcfile, dstfile))
    # 写入文件
    def write_data(self,filename, data, name):
        # 打开文件
        fo = open(filename, "a+")
        # 在文件末尾写入一行
        fo.seek(0, 2)
        fo.write(name)
        fo.write(str(data))
        fo.write("\n")
        # 关闭文件
        fo.close()


f1 = image_normalization()
srcfile = 'att_faces'
dstfile = 'normalization_faces_image'
f1.copyFile(srcfile, dstfile)