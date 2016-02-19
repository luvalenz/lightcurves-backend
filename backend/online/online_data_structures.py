__author__ = 'lucas'

import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import os


class OurMethodCluster:

    @property
    def radius(self):
        return self.sorted_distances[-1]

    @property
    def number_of_data(self):
        return len(self.sorted_data)

    def __init__(self, center, data_frame):
        self.center = np.array(np.matrix(center))
        self.sorted_distances = data_frame.loc[:,'distances'].values
        self.sorted_data = data_frame.drop('distances', 1).values
        self.sorted_data_ids = data_frame.index.values

    def get_ring_of_data(self, width):
        # if width <= 0:
        #     return self.sorted_data, self.sorted_data_ids
        ring_indices = np.where(self.sorted_distances > self.radius - width)[0]
        return self.sorted_data[ring_indices], self.sorted_data_ids[ring_indices]


class OurMethod:

    @property
    def number_of_distance_calculations(self):
        return self.number_of_step1_distance_calculations + self.number_of_step2_distance_calculations

    @property
    def number_of_features(self):
        return self.clusters_centers.shape[1]

    def __init__(self, name, clusters_path, simulation = True):
        self.name = name
        self.simulation = simulation
        self.similarity_function = OurMethod.euclidean_distance
        clusters_radii = pd.read_csv(os.path.join(clusters_path, self.name, 'radii.csv'), index_col=0).values
        self.clusters_radii = clusters_radii.reshape(clusters_radii.size,)
        self.number_of_clusters = len(self.clusters_radii)
        self.clusters_path = clusters_path
        self.clusters_centers = pd.read_csv(os.path.join(clusters_path, self.name, 'centers.csv'), index_col=0).values
        self.clusters = {}
        data_per_cluster = pd.read_csv(os.path.join(clusters_path, self.name, 'sizes.csv'), index_col=0).values
        self.data_per_cluster = data_per_cluster.reshape(data_per_cluster.size,)

    def query(self, target, k):
        self.number_of_step1_distance_calculations = 0
        self.number_of_step2_distance_calculations = 0
        self.number_of_data_after_filter = 0
        distances_target_to_centers = dist.cdist(np.array(np.matrix(target)), self.clusters_centers)[0]
       # distances_target_to_cluster_borders = distances_target_to_centers - self.clusters_radii
        self.number_of_step1_distance_calculations += len(self.clusters_centers)
        tau = 0
        n_searching_data = 0
        searching_clusters_indices = []
        i = 0
        while n_searching_data < k:
            closest_cluster_index = np.argpartition(distances_target_to_centers, i)[i] #ith smaller distance
            searching_clusters_indices.append(closest_cluster_index)
            distance_to_cluster = distances_target_to_centers[closest_cluster_index]
            if distance_to_cluster > tau:
                tau = distance_to_cluster
            number_of_data_in_cluster = self.get_cluster(closest_cluster_index).number_of_data
            n_searching_data += number_of_data_in_cluster
            i += 1
        searching_clusters_mask = np.zeros(self.number_of_clusters).astype(np.bool)
        searching_clusters_mask[searching_clusters_indices] = True
        ring_widths = self.clusters_radii + tau - distances_target_to_centers
        overlapping_clusters_mask = ring_widths > 0
        overlapping_clusters_indices = np.where(np.logical_and(np.logical_not(searching_clusters_mask), overlapping_clusters_mask))[0]
        searching_clusters_indices += overlapping_clusters_indices.tolist()
        self.number_of_disk_accesses = len(searching_clusters_indices)
        searching_data = np.empty((0, self.number_of_features))
        searching_data_ids = np.empty((0))
        for cluster_index in searching_clusters_indices:
            ring_width = ring_widths[cluster_index]
            cluster = self.get_cluster(cluster_index)
            data, data_ids = cluster.get_ring_of_data(ring_width)
            searching_data = np.vstack((searching_data, data))
            searching_data_ids = np.hstack((searching_data_ids, data_ids))
        self.number_of_data_after_filter = len(searching_data)
        return self.brute_force_search_vectorized(target, searching_data, searching_data_ids, k)


    def get_cluster(self, index):
        if not index in self.clusters:
            cluster_data_df = pd.read_csv(os.path.join(self.clusters_path, self.name, 'class{0}.csv'.format(index)), index_col=0)
            self.clusters[index] = OurMethodCluster(self.clusters_centers[index], cluster_data_df)
        return self.clusters[index]


    def brute_force_search_vectorized(self, target, candidates, candidates_ids, k):
        self.number_of_step2_distance_calculations += len(candidates)
        if not self.simulation:
            distances = dist.cdist(np.matrix(target), candidates)[0]
            order = distances.argsort()
            sorted_distances = distances[order]
            sorted_ids = candidates_ids[order]
            #print(sorted_ids)
            return sorted_ids[:k], sorted_distances[:k]

    def calculate_distance(self, v1, v2, step, number_of_operations=None, **kwargs):
        if number_of_operations is None:
            number_of_distance_calculations = 1
        else:
            number_of_distance_calculations = number_of_operations
        if step == 1:
            self.number_of_step1_distance_calculations += number_of_operations
        elif step == 2:
            self.number_of_step2_distance_calculations += number_of_operations
        axis = None
        if 'axis' in kwargs:
            axis = kwargs['axis']
        return self.similarity_function(v1,v2, axis)

    @staticmethod
    def euclidean_distance(v1, v2, axis = None):
        if axis is None:
            return np.linalg.norn(v2-v1)
        else:
            return np.linalg.norm(v2 - v1, axis=axis)
