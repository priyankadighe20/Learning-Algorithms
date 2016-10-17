import numpy as np
import matplotlib.pyplot as py
import math

# LOGISTIC REGRESSION USING NEWTONS METHOD
# Y = 0 for digit 3, Y = 1 for digit 5

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def calculate_hessian(w, x3, x5):
    H = np.zeros(shape=(64,64))

    T3 = len(x3)
    for t in range(T3):
        H += sigmoid(np.dot(w, x3[t].transpose())) * sigmoid(-1 * np.dot(w, x3[t].transpose())) * np.dot(x3[t].transpose(), x3[t])

    T5 = len(x5)
    for t in range(T5):
        H += sigmoid(np.dot(w, x5[t].transpose())) * sigmoid(-1 * np.dot(w, x5[t].transpose())) * np.dot(x5[t].transpose(), x5[t])

    return -1*H

def calculate_gradient(w, x3, x5):
    likelihood_gradient = np.zeros(shape=(64,1))

    # data points for Y = 0
    T3 = len(x3)
    for t in range(T3):
        error = -1 * sigmoid(np.dot(w, x3[t].transpose()))
        likelihood_gradient += error * x3[t].transpose()

    # data points for Y = 1
    T5 = len(x5)
    for t in range(T5):
        error = 1 - sigmoid(np.dot(w, x5[t].transpose()))
        likelihood_gradient += error * x5[t].transpose()

    return likelihood_gradient


def calculate_log_likelihood(w, x3, x5):
    log_likelihood = 0

    # data points for Y = 0
    T3 = len(x3)
    for t in range(T3):
        log_likelihood += math.log(sigmoid(-1 * np.dot(w, x3[t].transpose())))

    # data points for Y = 1
    T5 = len(x5)
    for t in range(T5):
        log_likelihood += math.log(sigmoid(np.dot(w, x5[t].transpose())))

    return log_likelihood

def get_file_data(path):
    X = []
    file = open(path)
    for line in file.readlines():
        line = line.strip('\n')
        line = line.split(" ")
        line.remove('')
        line = list(map(lambda s: int(s), line))
        X.append(line)

    return np.asmatrix(X)


def calculate_percent_error_rate(w, x3, x5):
    y3 = []
    y5 = []
    num_error3 = 0
    num_error5 = 0

    # data points for Y = 0
    T3 = len(x3)
    for t in range(T3):
        prediction = sigmoid(np.dot(w, x3[t].transpose()))
        if prediction > 0.5:
            y3.append(1)
            num_error3 += 1
        else:
            y3.append(0)

    # data points for Y = 1
    T5 = len(x5)
    for t in range(T5):
        prediction = sigmoid(np.dot(w, x5[t].transpose()))
        if prediction > 0.5:
            y5.append(1)
        else:
            y5.append(0)
            num_error5 += 1

    return 100*(num_error3/T3), 100*(num_error5/T5), 100 * ((num_error3+num_error5)/(T3+T5))


X3 = get_file_data("hw4_train3.txt")
X5 = get_file_data("hw4_train5.txt")

N = 10
W = np.zeros(shape=(1,64))

log_likelihood = []
p_error_rate = []

for i in range(N):
    H = calculate_hessian(W, X3, X5)
    dL = calculate_gradient(W, X3, X5)
    p_error_rate.append(calculate_percent_error_rate(W, X3, X5))
    P = np.dot(np.linalg.inv(H), dL)
    W = W - P.transpose()
    log_likelihood.append(calculate_log_likelihood(W, X3, X5))

print("The (log likelihood, percent error rate) for ", N, " iterations is: ")
for i in range(N):
    print(i, log_likelihood[i])

py.figure(1)
py.subplot(211)
py.plot(list(range(N)), log_likelihood)
py.xlabel("Number of iterations")
py.ylabel("log-likelihood")


print("The percent error rate for image3, image5 and combined for ", N, " iterations is: ")
for i in range(N):
    print(i,  p_error_rate[i])

py.subplot(212)
py.plot(list(range(N)), p_error_rate)
py.xlabel("Number of iterations")
py.ylabel("% error: 3, 5, both")
py.show()

print("\n The 8X8 weight matrix is : ")
print(W.reshape(8,8))

X3_test = get_file_data("hw4_test3.txt")
X5_test = get_file_data("hw4_test5.txt")

p_error_rate_test = calculate_percent_error_rate(W, X3_test, X5_test)


print("\nPercent error rate on the test data for image3, image5 and combined is : ", p_error_rate_test )


