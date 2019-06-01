#
# 这个Python代码的片段生成了3种不同类型的随机电势，
# 阶跃函数、分段线性函数和随机傅立叶级数。
# 这些类型的每一个都随着代的发展而变得更加“锯齿状”。
# 利用张量梯度求出各势的基态波函数。
# 薛定谔方程给出的能量泛函的下降法
# 将潜力和解决方案划分为训练和验证数据。
# 并将随机种子附加到文件名中并保存

import csv
import numpy as np
import tensorflow as tf
def subexp(expon):
    return np.power(abs(np.log(np.random.uniform())), expon)

def generatepot(style, param):        # 0 阶跃函数   1 分段线性函数 2 随机傅立叶级数  0-1“锯齿”刻度
    mu = 1. + bins*param              # 样式0＋1的平均跳跃点数
    forxp = 2.5 - 2*param             # 风格2的傅立叶指数
    scale = 5.0*(np.pi*np.pi*0.5)     # 能量标度

    if style < 2:
        dx = bins/mu
        xlist = [-dx/2]
        while xlist[-1] < bins:
            xlist.append(xlist[-1]+dx*subexp(1.))
        vlist = [scale*subexp(2.) for k in range(len(xlist))]
        k = 0
        poten = []
        for l in range(1, bins):
            while xlist[k+1] < l:
                k = k + 1
            if style == 0:
                poten.append(vlist[k])
            else:
                poten.append(vlist[k]+(vlist[k+1]-vlist[k])*(l-xlist[k])/(xlist[k+1]-xlist[k]))
    else:
        sincoef = [(2*np.random.randint(2)-1.)*scale*subexp(2.)/np.power(k, forxp) for k in range(1, bins//2)]
        coscoef = [(2*np.random.randint(2)-1.)*scale*subexp(2.)/np.power(k, forxp) for k in range(1, bins//2)]
        zercoef = scale*subexp(2.)
        poten = np.maximum(np.add(np.add(np.matmul(sincoef, sinval), np.matmul(coscoef, cosval)), zercoef), 0).tolist()
    return poten




seed = 18

np.random.seed(seed)

bins = 300    # 列
npots = 300
validnth = 5
'''
实际列数：bins-1
实际行数：3*npots/validnth  训练的数据行数
实际行数：3*npots*(validnth-1)/validnth  验证的数据行数
'''
sinval = np.sin([[np.pi*i*j/bins for i in range(1, bins)] for j in range(1, bins//2)])
cosval = np.cos([[np.pi*i*j/bins for i in range(1, bins)] for j in range(1, bins//2)])

sqrt2 = np.sqrt(2)  # 根号2

defgrdstate = tf.constant([sqrt2*np.sin(i*np.pi/bins) for i in range(1, bins)])

psi = tf.Variable(defgrdstate)

zerotens = tf.zeros([1])
psil = tf.concat([psi[1:], zerotens], 0)
psir = tf.concat([zerotens, psi[:-1]], 0)

renorm = tf.assign(psi, tf.divide(psi, tf.sqrt(tf.reduce_mean(tf.square(psi)))))

'''
梯度下降法是一个一阶最优化算法，通常也称为最速下降法。
要使用梯度下降法找到一个函数的局部极小值，
必须向函数上当前点对于梯度（或者是近似梯度）的反方向的规定步长距离点进行迭代搜索。
所以梯度下降法可以帮助我们求解某个函数的极小值或者最小值。
对于n维问题就最优解，梯度下降法是最常用的方法之一。
'''
optimzi = tf.train.GradientDescentOptimizer(0.0625/bins)

reinit = tf.assign(psi, defgrdstate)


init = tf.global_variables_initializer()

potentials = []
validpots = []
wavefuncs = []
validfuncs = []

'''
potentials 势能
validpots  验证势能
wavefuncs  波函数
validfuncs 验证博函数
'''
sess = tf.Session()

sess.run(init)

'''
minimize()
添加操作节点，用于最小化loss，并更新var_list
该函数是简单的合并了compute_gradients()与apply_gradients()函数
返回为一个优化更新后的var_list，如果global_step非None，该操作还会为global_step做自增操作
'''

for i in range(npots):
    if i % 10 == 0:
        print(str((100.*i)/npots) + '% complete')
    for j in range(3):

        vofx = generatepot(j, (1.*i)/npots)

        energy = tf.reduce_mean(tf.subtract(tf.multiply(tf.square(psi), tf.add(vofx, 1.*bins*bins)),
                                            tf.multiply(tf.multiply(tf.add(psil, psir), psi), 0.5*bins*bins)))
        training = optimzi.minimize(energy)
        sess.run(reinit)
        for t in range(30000):
            sess.run(training)
            sess.run(renorm)

        if i%validnth == 0:
            validpots.append(vofx)        # 验证势能
            validfuncs.append(sess.run(psi).tolist())   # 验证波函数
        else:
            potentials.append(vofx)   # 生成势能
            wavefuncs.append(sess.run(psi).tolist())  # 生成波函数

with open('files/test_pots'+str(seed)+'.csv', 'w', newline='') as f:
    fileout = csv.writer(f)
    fileout.writerows(potentials)


with open('files/valid_pots'+str(seed)+'.csv', 'w', newline='') as f:
    fileout = csv.writer(f)
    fileout.writerows(validpots)


with open('files/test_out'+str(seed)+'.csv', 'w', newline='') as f:
    fileout = csv.writer(f)
    fileout.writerows(wavefuncs)

with open('files/valid_out'+str(seed)+'.csv', 'w', newline='') as f:
    fileout = csv.writer(f)
    fileout.writerows(validfuncs)

print('Output complete')
