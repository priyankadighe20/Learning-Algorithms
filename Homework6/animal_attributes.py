__author__ = 'tanvinabar'

import numpy as np
import scipy as sc
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import defaultdict
from pylab import rcParams


def readAnimals():
    input = open("Animals_with_Attributes/classes.txt")
    data = []
    for l in input.readlines():
        data.append(l.strip().split("\t")[1])
    return data


def getPredicateMatrix():
    input = open("Animals_with_Attributes/predicate-matrix-continuous.txt")
    data = []
    for l in input.readlines():
        nums = l.strip().split(" ")
        nums = filter(None, nums)
        data.append([float(i) for i in nums])
    return data


def heirarchical_clustering (data):
    Z = linkage(data, "ward")
    plt.figure(figsize=(20, 20))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    plt.ylabel('Animal name')
    dendrogram( Z, leaf_rotation=120, leaf_font_size=8, orientation='right', labels = animals)
    plt.show()


def pca(data, n):
    pca = PCA(n_components=n)
    data_transform = pca.fit_transform(data)
    plt.figure(figsize=(25, 15))
    plt.title('PCA of Animal Set')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(column(data_transform, 0), column(data_transform, 1), alpha = 0.8, label = animals)
    for i, text in enumerate(animals):
        plt.annotate(text, xy= (data_transform[i][0], data_transform[i][1]), xytext = (data_transform[i][0]+2, data_transform[i][1]+2))
    plt.show()

def kmeans():
    k = 10
    kmeans = KMeans(init = 'random', n_clusters = k, n_init = k)
    kmeans.fit(pm)
    clusterdict = defaultdict(list)
    ctr = 0
    for l in kmeans.labels_:
        clusterdict[l].append(animals[ctr])
        ctr += 1

def column(matrix, i):
    return [row[i] for row in matrix]


animals = readAnimals()
pm = getPredicateMatrix()
kmeans()
heirarchical_clustering(pm)
pca(pm, 2)
