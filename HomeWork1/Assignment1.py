# Prototype Selection for MNIST database

import os, struct
from array import array as pyarray
import numpy as np
import random as rand

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


def calculate_error_rate(c_labels, t_labels):
    total_samples = len(t_labels)
    diff = np.subtract(c_labels, t_labels)
    error_samples = len(diff[diff != 0])
    print(error_samples, total_samples)
    return error_samples/total_samples


# def calculate_image_distance(test_image, train_image):
#     diff = test_image - train_image
#     sq_diff = diff * diff
#
#     return np.sqrt(np.sum(sq_diff))


def get_random_prototype_set(tr_images, tr_labels, m):
    # rand_index = np.random.choice(tr_images.shape[0], m, replace=False)
    # return tr_images[rand_index, :], tr_labels[rand_index]
    rs = rand.sample(list(zip(tr_images.tolist(), tr_labels.tolist())), m)
    return np.array([x[0] for x in rs]), np.array([x[1] for x in rs])


def get_cnn_prototype_set(tr_images, tr_labels, m):
    p_set_images = []
    p_set_labels = []
    tr_set = list(zip(tr_images.tolist(), tr_labels.tolist()))
    rand.shuffle(tr_set)

    while tr_set: # and len(p_set_images) < m:
        #rand.shuffle(tr_set)
        sample_set = tr_set.pop()
        if sample_set[1] != calculate_nearest_neighbor(np.array(sample_set[0]), np.array(p_set_images), np.array(p_set_labels)):
            p_set_images.append(sample_set[0])
            p_set_labels.append(sample_set[1])
            print("m =", len(p_set_images))

    return np.array(p_set_images), np.array(p_set_labels)


def calculate_nearest_neighbor(ts_image, tr_images, tr_labels):
    if len(tr_images) == 0 or len(train_labels) == 0:
        return -1

    test_image = np.array(ts_image)
    diff = np.array(tr_images - test_image)
    diff = [np.array(x.flatten()) for x in diff]
    distances = np.linalg.norm(diff, ord=2, axis=1)
    index = np.argmin(distances)
    nn = tr_labels[index]

    return nn


def classify_test_images(ts_images, tr_images, tr_labels):
    ts_labels = np.zeros(len(ts_images))
    i = 0
    for ts_image in ts_images:
        nn = calculate_nearest_neighbor(ts_image, tr_images, tr_labels)
        ts_labels[i] = nn
        print("Classify", i)
        i += 1

    return ts_labels



# Main program
print("Program started...")
print("Load started...")
train_images, train_labels = load_mnist("training")
test_images, test_labels = load_mnist("testing")
M = 10000

print("Get random prototype...")
random_proto_images, random_proto_labels = get_random_prototype_set(train_images, train_labels, M)
print("Classify...")
# test_images = np.array(test_images[:1000])
# test_labels = np.array(test_labels[:1000])
random_test_labels = classify_test_images(test_images, random_proto_images, random_proto_labels)
print("Calculate error rate...")
error_rate = calculate_error_rate(random_test_labels.flatten(), test_labels.flatten())
print("Random prototype error rate is :", error_rate)

print("Get cnn prototype...")
cnn_proto_images, cnn_proto_labels = get_cnn_prototype_set(train_images, train_labels, M)
print(len(cnn_proto_labels))
print("Classify...")
# test_images = np.array(test_images[:1000])
# test_labels = np.array(test_labels[:1000])
cnn_test_labels = classify_test_images(test_images, cnn_proto_images, cnn_proto_labels)
print("Calculate error rate...")
error_rate = calculate_error_rate(cnn_test_labels.flatten(), test_labels.flatten())
print("CNN prototype error rate is :", error_rate)
