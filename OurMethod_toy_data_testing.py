__author__ = 'lucas'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backend.offline.offline_interface import OfflineInterface
from backend.online.online_algorithms import OurMethod


def add_data_frame_to_database(db, df):
    index = df.index.values
    values = df.values
    list_of_dicts = []
    for i, v in zip(index, values):
        dictionary = {'reduced': list(v), 'id': str(i)}
        list_of_dicts.append(dictionary)
    db.add_many('test', list_of_dicts)


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


def plot_clusters(clusters):
    colors = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))
    np.random.shuffle(colors)
    for cluster, col in zip(clusters, colors):
        cluster_data = cluster.data_points
        plt.plot(cluster_data[:, 0], cluster_data[:, 1], 'o', markerfacecolor=col)
   # plt.plot(target[0], target[1], 'x')
    plt.show()

offline_interface = OfflineInterface()

# mean1 = [10, 10]
# mean2 = [20, 20]
# mean3 = [30, 30]
# mean4 = [40, 40]
# cov1 = [[3, 0], [0, 3]]
# cov2 = [[3, 0], [0, 3]]
# n = 2
# X1= np.random.multivariate_normal(mean1, cov1, n)
# X2= np.random.multivariate_normal(mean2, cov1, n)
# X3= np.random.multivariate_normal(mean3, cov1, n)
# X4 = np.random.multivariate_normal(mean4, cov2, n)
# X = np.vstack((X1, X2, X3, X4))
# # order = np.arange(len(X))
# # np.random.shuffle(order)
# # X = X[order]
# plt.plot(X[:, 0], X[:, 1], '*')
# plt.show()
# df = pd.DataFrame(X, index=[str(i) for i in range(len(X))])
# time_series_db = offline_interface.get_time_series_database()
# add_data_frame_to_database(time_series_db, df)
# brc = offline_interface.get_clustering_model()
# offline_interface.cluster_all()
# serialization_db = offline_interface.get_serialization_database()
# offline_interface.store_all_clusters()

clusters_db = offline_interface.get_clustering_database()
clusters = clusters_db.get_all()
target = [23.6, 26.3]
plot_clusters(clusters)
our_method = OurMethod(clusters_db)
ids, distances = our_method.query(target, 3)
print ids

time_series_db = offline_interface.get_time_series_database()
for id, distance in zip(ids, distances):
    ts = time_series_db.get_one('test', id)
    print ts.reduced_vector
    print ts.id
    print distance
    print np.linalg.norm(ts.reduced_vector - target)

