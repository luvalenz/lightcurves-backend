__author__ = 'lucas'

from backend.data_model.time_series import DataMultibandTimeSeries
from backend.data_model.time_series import MongoTimeSeriesDataBase
from backend.data_model.clusters import Cluster
from backend.offline.offline_algorithms import Birch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string


def extract_feature_matrix(database, id_list):
    time_series_iterator = database.get_many(id_list, 'macho', False)
    feature_vectors = []
    for time_series in time_series_iterator:
        feature_vector = time_series.reduced_vector
        if len(feature_vector) != 0:
            feature_vectors.append(feature_vector)
    #    else:
    #        print(time_series.id)
    #print('{0}, {1}'.format(len(id_list), len(feature_vectors)))
    return np.array((feature_vectors))


def plot_cluster_list(centers, clusters, database):
    plt.plot(centers[:, 0], centers[:, 1], 'x')
    colors = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))
    np.random.shuffle(colors)
    for cluster_indices, col in zip(clusters, colors):
        cluster_data = extract_feature_matrix(database, cluster_indices)
        plt.plot(cluster_data[:, 0], cluster_data[:, 1], 'o', markerfacecolor=col)
    plt.show()


def plot_lightcurves(lightcurve_list):
    reduced_features = []
    for lc in lightcurve_list:
        if lc.reduced_vector is not None and len(lc.reduced_vector) > 1:
            reduced_features.append(lc.reduced_vector)
    reduced_features = np.vstack(reduced_features)
    plt.plot(reduced_features[:, 0], reduced_features[:, 1], '*')
    plt.show()

mongodb = MongoTimeSeriesDataBase('lightcurves')
lightcurves_iterator = mongodb.find_many('macho', {}, False)

lightcurves = []

for lightcurve in lightcurves_iterator:
    lightcurves.append(lightcurve)

plot_lightcurves(lightcurves)

threshold = 0.75
remove_outliers = True
birch = Birch(threshold, remove_outliers, True, 10)
birch.add_many_time_series(lightcurves)


# local_centers, local_clusters = birch.get_cluster_list(mode='local')
# sizes = []
# for local_cluster in local_clusters:
#     sizes.append(len(local_cluster))
# print sizes
# # sorted_sizes = sorted(sizes)
# # sorted_sizes.reverse()
# # cum_sum = np.cumsum(sorted_sizes)
# # print(cum_sum)
# h= plt.hist(sizes, bins=101, cumulative=True)
# plt.show()
# print h
# # plt.plot(cum_sum)
# # sorted_sizes.reverse()
# # plt.xticks(sorted_sizes)
# # plt.show()



#print(len(local_centers))
#print(len(local_clusters))
#for cluster in local_clusters:
#    print str(len(cluster)) + ' ',
#print ' '
plot_cluster_list(local_centers, local_clusters, mongodb)

global_centers, global_clusters = birch.get_cluster_list(mode='global')
plot_cluster_list(global_centers, global_clusters, mongodb)


# clusters = []
# for i, center, cluster in zip(range(len(global_centers)),
#                                      global_centers, global_clusters):
#     time_series_list = mongodb.get_many(cluster, 'macho')
#     clusters.append(Cluster.from_time_series_sequence(time_series_list, center))