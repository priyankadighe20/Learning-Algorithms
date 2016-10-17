# Problem 2
import numpy as np
import matplotlib.pyplot as plt

# mu = np.array([0,0])
# sig = np.array([[9,0],[0,1]])
#
# x, y = np.random.multivariate_normal(mu, sig, 100).T
# plt.plot(x, y, 'x')
# plt.axis('equal')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Bi-variate Gaussian Distribution")
# plt.show()

# mu = np.array([0,0])
# sig = np.array([[1,-0.75],[-0.75,1]])
#
# x, y = np.random.multivariate_normal(mu, sig, 100).T
# plt.plot(x, y, 'o')
# plt.axis('equal')
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Bi-variate Gaussian Distribution")
# plt.show()

t_f = np.array([0, 0.05, 0.1, 0.15, 0.2])
x = np.array([10000, 9541, 9075, 8556, 8052])
a_f = (10000 - x)/10000
plt.plot(t_f, a_f)
plt.show()