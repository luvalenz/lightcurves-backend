from backend.offline.offline_algorithms import Birch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string

def add_data_frame(birch, df):
    index = df.index.values
    values = df.values
    for i, v in zip(index, values):
        birch._add_data_point(i, v)


def plot_clustering(centers, labels, unique_labels, X):
    plt.plot(centers[:, 0], centers[:, 1], 'x')
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for center, label in zip(centers, range(max(labels) + 1)) :
        #print center
        class_member_mask = (labels == label)
        X_class = X[class_member_mask]
        radius = 0
        for member in X_class:
            distance = np.linalg.norm(member - center)
            if distance > radius:
                radius = distance
        #print radius
        circle = plt.Circle(center,radius,color='r',fill=False)
        plt.gca().add_artist(circle)
    for label, col in zip(unique_labels, colors):
        class_member_mask = (labels == label)
        X_class = X[class_member_mask]
        plt.plot(X_class[:, 0], X_class[:, 1], 'o', markerfacecolor=col)
    plt.show()

def plot_cluster_list(centers, clusters, df):
    plt.plot(centers[:, 0], centers[:, 1], 'x')
    colors = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))
    np.random.shuffle(colors)
    for cluster_indices, col in zip(clusters, colors):
        cluster_data = df.loc[cluster_indices].values
        plt.plot(cluster_data[:, 0], cluster_data[:, 1], 'o', markerfacecolor=col)
    plt.show()

def test(birch, df):
    #print birch.count
    #print birch.is_fitted(mode='local')
    #print brc.is_fitted(mode='global')
    #print('')
    #print birch.get_number_of_clusters(mode='local')
    #print birch.get_number_of_clusters(mode='global')
    local_centers, local_clusters = birch.get_cluster_list(mode='local')
    global_centers, global_clusters = birch.get_cluster_list(mode='global')
    plot_cluster_list(local_centers, local_clusters, df)
    plot_cluster_list(global_centers, global_clusters, df)


mean1 = [10, 10]
mean2 = [20, 20]
mean3 = [30, 30]
mean4 = [40, 40]
mean5 = [50, 50]
cov1 = [[2.5, 0], [0, 2.5]]
cov2 = [[1, 0], [0, 1]]
n = 50
X1= np.random.multivariate_normal(mean1, cov1, n)
X2= np.random.multivariate_normal(mean2, cov1, n)
X3= np.random.multivariate_normal(mean3, cov1, n)
X4 = np.random.multivariate_normal(mean4, cov2, n)
X5 = np.random.multivariate_normal(mean5, cov2, n)
X6 = np.random.uniform(0, 50, 2*n).reshape((n,2))
X1_4 = np.vstack((X1, X2, X3, X4, X6))
order = np.arange(len(X1_4))
np.random.shuffle(order)
X1_4 = X1_4[order]
X = np.vstack((X1_4, X5))
#print X
# np.save('test_array', X)
plt.plot(X1_4[:, 0], X1_4[:, 1], '*')
plt.show()

df = pd.DataFrame(X, index=[hex(i) for i in range(len(X))])
#df = pd.DataFrame(X)
#print(df)


threshold = 3
outlier_rate = 1
brc = Birch(threshold, 'd1', 'r', 5, True, outlier_rate, 50)
add_data_frame(brc, df.iloc[0:n*5])
test(brc, df)

#### Incremental adding

add_data_frame(brc, df.iloc[n*5:])
test(brc, df)
