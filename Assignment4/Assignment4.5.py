import numpy as np
import math

def calculate_square_error(t, nasdaq, A):
    ft = nasdaq[t]
    for i in range(0,3):
        ft = ft - A[i]*nasdaq[t-(i+1)]

    pt = ft*ft

    return pt

def calculate_mean_square_error(nasdaq, A):
    square_error = 0
    T = len(nasdaq)
    for t in range(3, T):
        square_error += calculate_square_error(t, nasdaq, A)

    return square_error/(T-3)


# open and read nasdaq00.txt
file = open("nasdaq00.txt")
NASDAQ00 = file.readlines()
NASDAQ00 = list(map(lambda s: float(s.strip('\n')), NASDAQ00))
T0 = len(NASDAQ00)

# open and read nasdaq01.txt
file = open("nasdaq01.txt")
NASDAQ01 = file.readlines()
NASDAQ01 = list(map(lambda s: float(s.strip('\n')), NASDAQ01))
# We need to add last 3 days from 2000 to predict for the first day of 2001
NASDAQ01[0:0] = NASDAQ00[-3:]
T1 = len(NASDAQ01)

# Modelling the weight parameters
X = np.zeros(shape=(3,3))
Y = [0,0,0]
for t in range(3, T0):
    x = np.array([[NASDAQ00[t-1], NASDAQ00[t-2], NASDAQ00[t-3]]])
    prod = np.dot(x.transpose(), x)
    X = X + prod
    Y[0] = Y[0] + NASDAQ00[t] * NASDAQ00[t-1]
    Y[1] = Y[1] + NASDAQ00[t] * NASDAQ00[t-2]
    Y[2] = Y[2] + NASDAQ00[t] * NASDAQ00[t-3]

#print(X)
RHS = np.array([[Y[0]],[Y[1]],[Y[2]]])
A = np.linalg.solve(X, RHS)
print("The values for a1, a2, a3 are:\n", A)

print("Mean square error for year 2000 is: ", calculate_mean_square_error(NASDAQ00, A))
print("Mean square error for year 2001 is: ", calculate_mean_square_error(NASDAQ01, A))



