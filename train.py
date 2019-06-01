import matplotlib.pyplot as plt
import numpy as np
plt.figure()
x = np.arange(-10, 11)
y = np.log(np.exp(x)+1)
plt.title("Softplus")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x, y)
plt.show()