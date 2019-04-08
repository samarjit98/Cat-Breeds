import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc




def plot_clustering(X, labels, title=None):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("./plots/cluster")

# Visualizing dendogram
def plot_dendogram(X,linkage) :
    plt.figure(figsize=(10, 7))  
	plt.title("Customer Dendograms")  
    dend = shc.dendrogram(shc.linkage(X, method=linkage))  
    plt.savefig("./plots/dendogram")

#formation of clusters
def clusterformation(X) :
	for linkage in ('ward', 'average', 'complete', 'single'):
	    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    	t0 = time()
    	clustering.fit(X)
    	plot_clustering(X, clustering.labels_, "%s linkage" % linkage)
    	plot_dendogram(X,linkage)



