# Gibbs Sampler
# from scipy.misc import logsumexp
import numpy as np
import random

def parse_train_data_file(filename):
    data = np.zeros(shape=(D, V))

    file = open(filename, "r")
    for line in file.readlines():
        line = (line.strip('\n')).split(' ')
        line = list(map(lambda s: int(s), line))
        data[line[0]-1][line[1] - 1] += line[2]
    file.close()

    return data


def sample_from_discrete(probs):
    temp = random.random()
    total = 0
    for i in range(len(probs)):
        total += probs[i]
        if temp < total:
            return i;
    print("Failure!")
    return -1


V = 61188
D = 1633
K = 3
# Parsing the training data
TR_DATA = parse_train_data_file(r"data291/train.data")

ALPHA = 0.1 * np.ones(shape=(K, V))
BETA = np.ones(K)
PI = np.random.dirichlet(BETA)
THETA = np.ones(shape=(K, V))
for k in range(K):
    THETA[k] = np.random.dirichlet(ALPHA[k])

Z = np.array([sample_from_discrete(PI) for d in range(D)])

for itr in range(200):
    PI = np.random.dirichlet(BETA + np.bincount(Z, minlength=K))
    DATA = np.zeros(shape=(K, V))
    for d in range(D):
        DATA[Z[d]] += TR_DATA[d]

    for k in range(K):
        THETA[k] = np.random.dirichlet(ALPHA[k] + DATA[k])

    probs = np.log(PI)
    for k in range(K):
        probs[k] += np.sum(np.multiply(DATA[k], np.log(THETA[k])))

    c = np.max(probs)
    probs -= c
    x = np.exp(probs)
    Z = np.array([sample_from_discrete(x/(np.sum(x))) for d in range(D)])
    # print(Z)

for k in range(K):
    w_i = np.argsort(THETA[k])[:10]
    print(w_i)