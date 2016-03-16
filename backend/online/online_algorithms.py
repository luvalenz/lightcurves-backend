__author__ = 'lucas'

import numpy as np
import scipy.spatial.distance as dist


class OurMethod:

    @property
    def number_of_distance_calculations(self):
        return self.number_of_step1_distance_calculations + self.number_of_step2_distance_calculations

    @property
    def number_of_features(self):
        return self.clusters_centers.shape[1]

    @property
    def clusters_radii(self):
        return self._clusters_db.radii

    @property
    def clusters_centers(self):
        return self._clusters_db.centers

    @property
    def clusters_counts(self):
        return self._clusters_db.counts

    @property
    def clusters_ids(self):
        return self._clusters_db.cluster_ids

    @property
    def number_of_clusters(self):
        return len(self.clusters_counts)

    def __init__(self, clusters_db, time_series_db, simulation=False):
        self.simulation = simulation
        self.similarity_function = OurMethod.euclidean_distance
        self._clusters_db = clusters_db
        self._time_series_db = time_series_db

    def time_series_query(self, target, k):
        reduced_vector = target.reduced_vector
        retrieved_ids, retrieved_distances = self.vector_query(reduced_vector, k)
        retrieved_time_series = self._time_series_db.get_many(retrieved_ids, None, False)
        return retrieved_time_series, retrieved_distances

    def vector_query(self, target, k):
        self.number_of_step1_distance_calculations = 0
        self.number_of_step2_distance_calculations = 0
        self.number_of_data_after_filter = 0
        distances_target_to_centers = dist.cdist(np.array(np.matrix(target)), self.clusters_centers)[0]
        cluster_order_by_distance = np.argsort(distances_target_to_centers)
        self.number_of_step1_distance_calculations += len(self.clusters_centers)
        tau = 0
        n_searching_data = 0
        searching_clusters_indices = []
        i = 0
        while n_searching_data < k:
            closest_cluster_index = cluster_order_by_distance[i] #ith closest cluster
            searching_clusters_indices.append(closest_cluster_index)
            distance_to_cluster = distances_target_to_centers[closest_cluster_index]
            cluster_radius = self.clusters_radii[closest_cluster_index]
            new_tau = distance_to_cluster + cluster_radius
            if new_tau > tau:
                tau = new_tau
            number_of_data_in_cluster = self.clusters_counts[closest_cluster_index]
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
        return self._clusters_db.get_cluster(self.clusters_ids[index])

    def brute_force_search_vectorized(self, target, candidates, candidates_ids, k):
        self.number_of_step2_distance_calculations += len(candidates)
        if not self.simulation:
            distances = dist.cdist(np.matrix(target), candidates)[0]
            order = distances.argsort()
            sorted_distances = distances[order]
            sorted_ids = candidates_ids[order].tolist()
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
