import math
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.log(np.cosh(x))


def df(x):
    return np.tanh(x)


def d2f(x):
    return 1 - np.power(np.tanh(x),2)


def g(x):
    sum = 0
    for k in range(10):
        sum += np.log(np.cosh(x + 1/np.sqrt(k+1)))

    return sum/10

def dg(x):
    sum = 0
    for k in range(10):
        sum += np.tanh(x + 1/np.sqrt(k+1))

    return sum/10


def Q(x,y):
    return f(y) + df(y)*(x-y) + (0.5 * np.power(x-y, 2))


def update_x_aux(xn):
    a = []

    print("\nThe values of xn and f(xn) using auxiliary function with xn = ", xn, " are:")

    for i in range(10):
        xn = xn - df(xn)
        a.append(xn)
        print(xn, f(xn))

    return a

def update_x_newton(xn):
    a = []
    print("\nThe values of xn and f(xn) using Newton's method with xn = ", xn, " are:")

    for i in range(10):
        xn = xn - (df(xn)/d2f(xn))
        a.append(xn)
        print(xn, f(xn))

    return a

def update_x_aux_g(xn):
    a = []
    print("\nThe values of xn and g(xn) using auxiliary function with xn = ", xn, " are:")

    for i in range(30):
        xn = xn - dg(xn)
        a.append(xn)
        print(xn, g(xn))

    return a


x = np.linspace(-20, 20, 1000)
plt.plot(x, f(x))
plt.plot(x, Q(x, -2))
plt.plot(x, Q(x, 3))
plt.xlabel("x")
#plt.show()


plt.plot(x, g(x))
plt.xlabel("x")
plt.ylabel("g(x)")
#plt.show()

X = update_x_aux(-2)
X1 = update_x_aux(3)
plt.plot(X)
plt.plot(X1)
plt.xlabel("n (number of iterations)")
plt.ylabel("xn for f(x)")
#plt.show()

X = update_x_newton(1)
plt.plot(X)
plt.xlabel("n (number of iterations)")
plt.ylabel("xn for newton)")
#plt.show()

X = update_x_aux_g(3)
plt.plot(X)
plt.xlabel("n (number of iterations)")
plt.ylabel("xn for g(x)")
#plt.show()
