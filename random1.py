import random
num = 0;
for i in range(0,1000000):
    if random.randint(0, 1) == 0:
        num += 1

if num/1000000>0.5:
    print('不分手')
else :
    print('分手')

