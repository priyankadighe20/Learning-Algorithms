import os, struct
import numpy as np
from array import array as pyarray
from matplotlib import pyplot as plt
from copy import copy, deepcopy

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


def nbr(x, y):
    nset = set([(x, y-1), (x-1,y), (x+1, y), (x, y+1)])
    if x==0:
        nset = nset - set([(x-1, y)])
    if y==0:
         nset = nset - set([(x, y-1)])
    if x==27:
        nset = nset - set([(x+1, y)])
    if y==27:
         nset = nset - set([(x, y+1)])

    return nset


def update(mu, x, y, b):
    val = 0
    for n in nbr(x, y):
        val = val + mu[n[0]][n[1]]

    val += b[x][y]

    return np.tanh(val)

# gives the probability of img[x][y] = 1
def update_g(img, x, y, b):
    n = nbr(x, y)
    val = 0
    for n in nbr(x,y):
        val += img[n[0]][n[1]]

    val += b[x][y]

    p1 = 1/(1 + np.exp(val * -2))
    p[x][y] = p1

    if p1 > 0.5:
        return 1
    else:
        return -1


def calculate_error(g_image, o_image):
    diff = g_image - o_image
    return np.linalg.norm(diff)


def mean_field_upd():
    mu = deepcopy(image)
    plt.imshow(mu, interpolation='nearest')
    plt.show()

    for itr in range(30):
        for x in range(28):
            for y in range(28):
                mu[x][y] = update(mu, x, y, b)

        if itr%5 == 0:
            print itr, calculate_error(mu, o_image)
            # print iter, calculate_error(image, o_image)

    plt.imshow(mu, interpolation='nearest')
    plt.show()


def gibbs_update():
    plt.imshow(image, interpolation='nearest')
    plt.show()
    for itr in range(30):
        for x in range(28):
            for y in range(28):
                image[x][y] = update_g(image, x, y, b)

        if itr%5 == 0:
            # plt.imshow(p, interpolation='nearest')
            # plt.show()
            print itr, calculate_error(image, o_image)

    plt.imshow(p, interpolation='nearest')
    plt.show()

    plt.imshow(image, interpolation='nearest')
    plt.show()


images, _ = load_mnist("training", [1])
# Pick an image from the training images
image_copy = deepcopy(images[0])
image = np.ones(shape=(28,28))

for i in range(28):
    for j in range(28):
        if image_copy[i][j] == 0:
            image[i][j] = -1

# Randonly flip 10% of the bits
samplesize = int(np.size(image) * 0.1)
x_flip = np.random.choice(28, samplesize, replace=True)
y_flip = np.random.choice(28, samplesize, replace=True)

o_image = deepcopy(image)

for i in range(samplesize):
    if image[x_flip[i]][y_flip[i]] == -1:
        image[x_flip[i]][y_flip[i]] = 1
    elif image[x_flip[i]][y_flip[i]] == 1:
        image[x_flip[i]][y_flip[i]] = -1

print calculate_error(image, o_image)

c = 2
b = image * c
p = np.zeros(shape=(28,28))


mean_field_upd()
# gibbs_update()
