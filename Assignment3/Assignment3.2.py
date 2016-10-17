import random
import math
import matplotlib.pyplot as py
import numpy as np

def LIKELIHOOD_WEIGHT(b):
    return (0.65/1.35)*(0.35**math.fabs(Z-b))

def GENERATE_B(n):
    num = 0;
    for i in range(0,n):
        num = num << 1
        num += random.randint(0, 1)

    return num


def PR_QUERY_W_EVIDENCE(N):
    countBi = 0
    totalCount = 0
    probArray=[]
    for i in range(0,N):
        B = GENERATE_B(10) # Generate a random B
        lw = LIKELIHOOD_WEIGHT(B)
        totalCount = totalCount + lw
        if B >= 2**6: # That means that ith bit is set
            countBi = countBi + lw
        prob = countBi/totalCount
        probArray.append(prob)

    return probArray, prob




Z = 64
X = 1000000
probArray, finalProb = PR_QUERY_W_EVIDENCE(10**7)
print("Probability for 10^7 samples is ", finalProb)

probArray, finalProb = PR_QUERY_W_EVIDENCE(X)
print(finalProb)
py.plot(list(range(X)), probArray )
py.show()