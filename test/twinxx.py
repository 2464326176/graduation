#!/usr/bin/python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 20, 1)
y1 = x * x
y2 = np.log(x)

fig = plt.figure()
ax1 = fig.add_subplot(111)
# print(ax1)
ax1.plot(x, y1, label="$y1 = x * x$", color="r")
ax1.legend(loc=0)
# 设置对应坐标轴的名称
ax1.set_ylabel("y1")
ax1.set_xlabel("Compare y1 and y2")

# 设置x轴刻度的数量
ax = plt.gca()
ax.locator_params("x", nbins=20)

# 添加坐标轴,并在新添加的坐标轴中画y2 = log(x)图像
ax2 = plt.twinx()
ax2.set_ylabel("y2")
ax2.plot(x, y2, label="$y2 = log(x)$")
ax2.legend(loc=0)

plt.show()

# 也可以设置两个x轴，方法和双y轴相同，要把plot中对应的x和y互换，这样显示的结果和双y轴基本相同