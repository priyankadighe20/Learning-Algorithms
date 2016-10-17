# Assignment 5.2
import numpy as np
import math


def noisy_or_cpt_y0(p, x):
    val = 1
    for i in range(len(x)):
        val *= math.pow(1-p[i], x[i])

    return val


def noisy_or_cpt_y1(p, x):
    return 1 - noisy_or_cpt_y0(p, x)


def update_Pi(p, i, count, X, y):
    sum = 0
    T = len(y)
    for t in range(T):
        sum += (y[t] * X[t][i] * p[i]) / (noisy_or_cpt_y1(P, X[t]))

    return sum/count[i]


def update_P(P, count, X, y):
    n = len(X[0])
    temp_p = n*[0]
    for i in range(n):
        temp_p[i] = update_Pi(P, i, count, X, y)

    return temp_p

def calculate_log_likelihood(p, X, y):
    T = len(Y)
    sum = 0

    for t in range(T):
        a0 = noisy_or_cpt_y0(p, X[t])
        sum += y[t]*math.log(1-a0) + (1-y[t])*math.log(a0)

    return sum/T


X =[]
file = open("hw5_X.txt")
lines = file.readlines()
for line in lines:
    line = line.strip('\n')
    line = list(map(lambda s: int(s), line.split()))
    X.append(line)

Y = []
file = open("hw5_Y.txt")
Y = file.readlines()
Y = list(map(lambda s: int(s.strip('\n')), Y))


N = 64  # Number of iterations
P = len(X[0])*[0.5]
COUNT = np.sum(X, axis=0)

L = []

L.append(calculate_log_likelihood(P, X, Y))

for iteration in range(64):
    P = update_P(P, COUNT, X, Y)
    L.append(calculate_log_likelihood(P, X, Y))

print(L[0], L[1], L[2], L[4], L[8], L[16], L[32])