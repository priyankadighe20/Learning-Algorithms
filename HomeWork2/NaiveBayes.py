# CSE 250B Homework 2
import numpy as np
import math
import gc


def parse_label_file(filename):
    file = open(filename, "r")
    lines = file.readlines()
    lines = list(map(lambda s: int(s.strip('\n')), lines))
    file.close()

    return np.array(lines)


def parse_train_data_file(filename):
    data = np.zeros(shape=(L, V))

    file = open(filename, "r")
    for line in file.readlines():
        line = (line.strip('\n')).split(' ')
        line = list(map(lambda s: int(s), line))
        data[TR_LABEL[line[0]-1] - 1][line[1] - 1] += line[2]
    file.close()

    return data


def parse_test_data_file(filename):
    gc.collect()
    data = np.zeros(shape=(D, V))

    file = open(filename, "r")
    for line in file.readlines():
        line = (line.strip('\n')).split(' ')
        line = list(map(lambda s: int(s), line))
        if line[0] <= D:
            data[line[0]-1][line[1]-1] = line[2]

    return data


def parse_test_data_and_classify(filename):
    data = np.zeros(shape=(1, V))

    file = open(filename, "r")
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = (lines[i].strip('\n')).split(' ')
        lines[i] = tuple(list(map(lambda s: int(s), lines[i])))

    print(lines)
    return data


def calculate_error_rate(c_label, t_label):
    diff = np.subtract(c_label, t_label)

    error_rate = np.count_nonzero(diff) / len(t_label)
    return error_rate * 100


def calculate_prob_given_y(document, y):
    px = math.log(PI[y])
    total_x = np.sum(TR_DATA[y])

    px += np.dot(document, np.log((TR_DATA[y] + 1) / total_x))

    return px


def classify_document(ts_data, doc_id):
    py = np.zeros(shape=(L, 1))

    for j in range(L):
        py[j] = calculate_prob_given_y(ts_data[doc_id], j)

    return np.argmax(py) + 1


def classify_test_data(ts_data):
    d = len(ts_data)
    c_label = np.zeros(shape=(d, 1))

    for i in range(d):
        c_label[i] = classify_document(ts_data, i)

    return c_label

V = 61188
L = 20
D = 1000
# Parsing the training data
# Labels are from 0 to 19
TR_LABEL = parse_label_file(r"data/train.label")
TR_DATA = parse_train_data_file(r"data/train.data")

# Calculate PI array
PI = np.bincount(TR_LABEL)[1:]
PI = PI/PI.sum()

# Parsing the training data
TS_LABEL = parse_label_file(r"data/test.label")
TS_DATA = parse_test_data_file(r"data/test.data")
# C_LABEL = parse_test_data_and_classify(r"data/test.data")

C_LABEL = classify_test_data(TS_DATA)

print(calculate_error_rate(C_LABEL.flatten(), TS_LABEL.flatten()[:D]))

