import os
import cv2
import numpy as np

import sys

c = 0
sz=None
X,y = [], []

for root, dirs, files in os.walk("ly"):
    for name in files:
        try:
            if (name == ".directory"):
                continue
            filepath = os.path.join(root, name)

            im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if (im is None):
                print("image " + filepath + " is none")
            else:
                print(filepath)
            # resize to given size (if given)
            if (sz is not None):
                im = cv2.resize(im, (200, 200))

            X.append(np.asarray(im, dtype=np.uint8))
            y.append(c)
        except IOError:
            print("I/O error({0}): {1}".format(errno, strerror))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
        c = c+1
print([X,y])