import random
import math
# import matplotlib.pyplot as py
# import numpy as np

def LIKELIHOOD_WEIGHT(b):
    return (0.65/1.35)*(0.35**math.fabs(Z-b))

def GENERATE_B(n):
    num = 0;
    for i in range(0,n):
        num = num << 1
        num += random.randint(0, 1)

    return num


def PR_QUERY_W_EVIDENCE(N):
    nr = 0
    dr = 0
    probArray=[]
    for i in range(0,N):
        B = GENERATE_B(10) # Generate a random B
        lw = LIKELIHOOD_WEIGHT(B)
        dr = dr + lw
        if B >= 2**6:
            nr = nr + lw
        prob = nr/dr
        probArray.append(prob)

    return probArray, prob




Z = 64
#PR_QUERY_W_EVIDENCE(10**7)

# X = 1000000
#
# probArray, finalProb= PR_QUERY_W_EVIDENCE(X)
# print(finalProb)
# py.plot(list(range(X)), probArray )
# py.show()