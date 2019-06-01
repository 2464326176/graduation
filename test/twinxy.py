# !/usr/bin/python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 21, 0.1)

y1 = x * x
y2 = np.log(x)

plt.plot(x, y1)

# 添加一条坐标轴，y轴的
plt.twinx()
plt.plot(x, y2)

plt.show()
