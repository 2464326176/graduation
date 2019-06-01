import csv
import matplotlib.pyplot as plt
import time
import os
import numpy as np
def output(filename):
    with open(filename, newline='') as csvfile:
        result = []
        reader = csv.reader(csvfile)
        reader = csv.reader(csvfile)
        for row in reader:
            result.append([float(num) for num in row])
        return result
for i in range(180):
    out = output('valid_out17.csv')
    pots = output('valid_pots17.csv')
    plt.figure(str(i))
    plt.subplot(211)

    plt.plot(pots[i])
    plt.ylabel('势能')
    plt.subplot(212)
    plt.plot(out[i])
    plt.ylabel('实际波函数')

    if os.path.exists('valid') == False:
        os.mkdir('valid')
    plt.savefig('valid/' + time.strftime("%Y%m%d%H%M%S__") + '__' + str(i) + '.png')
    # plt.show()


