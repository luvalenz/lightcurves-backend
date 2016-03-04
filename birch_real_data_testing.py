__author__ = 'lucas'

from backend.data_model.time_series import DataMultibandTimeSeries
from backend.data_model.time_series import TimeSeriesMongoDataBase
from backend.offline.offline_algorithms import Birch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string


def extract_feature_matrix(database, id_list):
    time_series_list = database.get_many('macho', id_list)
    feature_vectors = []
    for time_series in time_series_list:
        feature_vector = time_series.reduced_vector
        if len(feature_vector) != 0:
            feature_vectors.append(feature_vector)
        else:
            print(time_series.id)
    print('{0}, {1}'.format(len(id_list), len(feature_vectors)))
    return np.array((feature_vectors))


def plot_cluster_list(centers, clusters, database):
    plt.plot(centers[:, 0], centers[:, 1], 'x')
    colors = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))
    for cluster_indices, col in zip(clusters, colors):
        cluster_data = extract_feature_matrix(database, cluster_indices)
        plt.plot(cluster_data[:, 0], cluster_data[:, 1], 'o', markerfacecolor=col)
    plt.show()

mongodb = TimeSeriesMongoDataBase('lightcurves')
lightcurves = mongodb.find_many('macho', {})

threshold = 3
birch = Birch(threshold, 'd1', 'r', 4)

birch.add_many_time_series(lightcurves)


local_centers, local_clusters = birch.get_cluster_list(mode='local')
print(len(local_centers))
print(len(local_clusters))
for cluster in local_clusters:
    print str(len(cluster)) + ' ',
print ' '
plot_cluster_list(local_centers, local_clusters, mongodb)

global_centers, global_clusters = birch.get_cluster_list(mode='global')
plot_cluster_list(global_centers, global_clusters, mongodb)