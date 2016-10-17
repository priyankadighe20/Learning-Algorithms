import os, struct
from array import array as pyarray
import numpy as np
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp
from operator import itemgetter
import random
import matplotlib.pyplot as plt


def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.uint8)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return np.array(images), np.array(labels)


def create_GaussianModel(tr_images):
    m = len(tr_images)
    mu = np.sum(tr_images, axis=0)/m

    sig = np.cov(tr_images.T)
    sig = sig + (C * np.identity(784))

    GAUSS.append(multivariate_normal(mean=mu, cov=sig))


def calculate_error_rate(c_labels, t_labels):
    total_samples = len(t_labels)
    diff = np.subtract(c_labels, t_labels)
    error_samples = len(diff[diff != 0])
    print(error_samples, total_samples)
    return error_samples/total_samples


def classify_images(ts_images):
    ts_labels = np.zeros(len(ts_images))
    k = 0
    for x in ts_images:
        px = np.zeros(10)
        for i in range(10):
            px[i] = PI[i] + GAUSS[i].logpdf(x)

        ts_labels[k] = np.argmax(px)
        k += 1

    return ts_labels


def classify_images_abstaining(ts_images, ts_labels, t_list):
    data_set = []
    k = 0
    for x in ts_images:
        px = np.zeros(10)
        for i in range(10):
            px[i] = PI[i] + GAUSS[i].logpdf(x)

        # Find the diff between 2 max values and inserting the tuple in a set
        c_label = np.argmax(px)
        max1, max2 = find_two_max(px)
        diff = max1 - max2
        data_set.append((diff, c_label, ts_labels[k]))
        k += 1

    # Now sort the data set wrt diff
    data_set = sorted(data_set, key=itemgetter(0), reverse=True)
    d, c_labels, t_labels = zip(*data_set)
    error_rate = []
    for t in t_list:
        sp = np.argmax(np.array(d) < t)
        if sp == 0 and d[sp] > t:
            sp = len(d)
        print("Split at i = ", sp)
        error_rate.append(calculate_error_rate(np.array(c_labels[:sp]).flatten(), np.array(t_labels[:sp]).flatten()))

    print("Plotting error rate with f...")
    plt.plot(F, error_rate)
    plt.show()


def calculate_threshold_from_val_data(ts_images, ts_labels, f_list):
    data_set = []
    k = 0
    for x in ts_images:
        px = np.zeros(10)
        for i in range(10):
            px[i] = PI[i] + GAUSS[i].logpdf(x)

        # Find the diff between 2 max values and inserting the tuple in a set
        c_label = np.argmax(px)
        max1, max2 = find_two_max(px)
        diff = max1 - max2
        data_set.append((diff, c_label, ts_labels[k]))
        k += 1

    # Now sort the data set wrt diff
    data_set = sorted(data_set, key=itemgetter(0), reverse=True)

    # For each f we need to find a threshold t
    t_list = []
    for f in f_list:
        sp = int(len(data_set)*(1-f))
        t_list.append(data_set[sp-1][0])

    print(t_list)
    return t_list


def find_two_max(p):
    p = sorted(p, reverse=True)
    return p[0], p[1]


def posterior(x):
    px = np.zeros(10)
    for i in range(10):
        px[i] = PI[i] + GAUSS[i].logpdf(x)

    log_denominator = logsumexp(px)
    return np.exp(px - log_denominator)


def display_misclassified(ts_images, c_labels, t_labels):
    diff = np.subtract(c_labels, t_labels)
    sample = np.nonzero(diff)
    r_sample = random.sample(sample[0].tolist(), 5)

    print("Printing misclassified labels")
    for s in r_sample:
        print(t_labels[s], c_labels[s], posterior(ts_images[s]))


# Main program
print("Program started...")
print("Load started...")

# Create array to save mu and sigma matrices
GAUSS = []
C = 3100
VAL_I = []
VAL_LB = []
PI = np.zeros(10)
F = [0, 0.05, 0.1, 0.15, 0.2]

# Started Loading Data
for i in range(10):
    train_images, train_labels = load_mnist("training", digits=[i])
    train_images = train_images.reshape(len(train_images), 784)
    np.random.shuffle(train_images)
    split = int(len(train_images) * 0.8)
    x_vector = train_images[:split]
    VAL_I.extend(train_images[split:])
    VAL_LB.extend(train_labels[split:])
    PI[i] = len(x_vector)
    create_GaussianModel(x_vector)

PI = np.log(PI/np.sum(PI))
VAL_I = np.array(VAL_I)
VAL_LB = np.array(VAL_LB).flatten()

print("Validate Data...")
labels = classify_images(VAL_I)
print("Error rate on validation set is : ", calculate_error_rate(labels.flatten(), VAL_LB))

# print("Calculate abstaining threshold from validation data...")
# THRESHOLD = calculate_threshold_from_val_data(VAL_I, VAL_LB, F)
# print("Error rate on validation set is : ", calculate_error_rate(labels.flatten(), VAL_LB))

# Classify test images on the basis of Gaussian
# print("Classify Data...")
test_images, test_labels = load_mnist("testing")
test_images = test_images.reshape(len(test_images), 784)
# labels = classify_images(test_images)
# print("Error rate on test set is : ", calculate_error_rate(labels.flatten(), test_labels.flatten()))

# display_misclassified(test_images, labels.flatten(), test_labels.flatten())

# Classify the data by abstaining
# print("Classify test data by abstaining...")
# classify_images_abstaining(test_images, test_labels, THRESHOLD)