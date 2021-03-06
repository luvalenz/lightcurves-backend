__author__ = 'lucas'

import numpy as np
import scipy.spatial.distance as dist
import time

class OurMethod:

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
        self._clusters_db = clusters_db
        self._time_series_db = time_series_db

    def time_series_query(self, target, k):
        reduced_vector = target.reduced_vector
        retrieved_ids, retrieved_distances, step1_calc_time, step2_calc_time, \
            fetch_time, n_data_after_filter, n_visited_clusters = self.vector_query(reduced_vector, k)
        retrieved_time_series = self._time_series_db.get_many(retrieved_ids, None, False, None, True)
        if self.simulation:
            return retrieved_time_series, retrieved_distances, step1_calc_time,  \
                step2_calc_time, fetch_time, n_data_after_filter, n_visited_clusters
        return retrieved_time_series, retrieved_distances

    def vector_query(self, target, k):
        clusters_centers = self.clusters_centers
        time0 = time.time()
        distances_target_to_centers = dist.cdist(np.array(np.matrix(target)), clusters_centers)[0]
        cluster_order_by_distance = np.argsort(distances_target_to_centers)
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
        overlapping_clusters_indices = \
            np.where(np.logical_and(np.logical_not(searching_clusters_mask), overlapping_clusters_mask))[0]
        searching_clusters_indices += overlapping_clusters_indices.tolist()
        searching_data = np.empty((0, self.number_of_features))
        searching_data_ids = np.empty((0))
        time1 = time.time()
        for cluster_index in searching_clusters_indices:
            ring_width = ring_widths[cluster_index]
            cluster = self.get_cluster(cluster_index)
            data, data_ids = cluster.get_ring_of_data(ring_width)
            searching_data = np.vstack((searching_data, data))
            searching_data_ids = np.hstack((searching_data_ids, data_ids))
        time2 = time.time()
        result_ids, result_distances = self.brute_force_search_vectorized(target, searching_data, searching_data_ids, k)
        time3 = time.time()
        number_of_data_after_filter = len(searching_data)
        number_of_visited_clusters = len(searching_clusters_indices)
        step1_calc_time = time1 - time0
        fetching_time = time2 - time1
        step2_calc_time = time3 - time2
        return result_ids, result_distances, step1_calc_time,\
               step2_calc_time, fetching_time, number_of_data_after_filter, number_of_visited_clusters

    def peek_cluster(self, index):
        return self._clusters_db.peek_cluster(self.clusters_ids[index])

    def get_cluster_from_cursor(self, cursor):
        return self._clusters_db.get_cluster_from_cursor(cursor)

    def get_cluster(self, index):
        return self._clusters_db.get_cluster(self.clusters_ids[index])

    def brute_force_search_vectorized(self, target, candidates, candidates_ids, k):
        distances = dist.cdist(np.matrix(target), candidates)[0]
        order = distances.argsort()
        sorted_distances = distances[order]
        sorted_ids = candidates_ids[order].tolist()
        return sorted_ids[:k], sorted_distances[:k]

