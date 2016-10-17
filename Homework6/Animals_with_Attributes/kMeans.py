import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from pylab import rcParams
rcParams['figure.figsize'] = 50, 100


def load_classes(filename):
    file = open(filename)
    classes = file.readlines()
    classes = list(map(lambda s: s.strip('\n'), classes))
    classes = list(map(lambda s: s.strip(' ').split('\t'), classes))
    classes = np.array(classes)

    return classes[:, 1]


def load_data(filename):
    file = open(filename)
    data = []
    for line in file.readlines():
        line = line.strip('\n')
        line = line.strip(' ').split()
        line = list(map(lambda s: float(s), line))
        data.append(line)

    return np.array(data)


def k_means(data, classes, k):
    c_pred = KMeans(n_clusters=k, n_init=5, init='k-means++').fit_predict(data)
    clusters = sorted(list(zip(c_pred, classes)), key=lambda x: x[0])
    print(clusters)


def hierarchical_linkage(data, classes):
    c_pred = linkage(data, 'average')

    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(c_pred, orientation='right', labels=classes)
    plt.show()


def projection(data, classes):
    proj_data = PCA(n_components=2).fit_transform(data)
    plt.scatter(proj_data[:,0], proj_data[:, 1], marker="o", alpha=0.8, label=classes)

    for i, txt in enumerate(classes):
        plt.annotate(txt, (proj_data[i, 0], proj_data[i, 1]))

    plt.title("PCA on Animals Set")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def main():
    CLASSES = load_classes("classes.txt")
    K = 10
    DATA = load_data("predicate-matrix-continuous.txt")

    # KMeans on the data
    k_means(DATA, CLASSES, K)

    # Hierarchical clustering using average linkage
    hierarchical_linkage(DATA, CLASSES)

    projection(DATA, CLASSES)

main()