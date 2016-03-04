__author__ = 'lucas'

import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
import itertools
import pandas as pd
import scipy.spatial.distance as dist
import os
import pickle
import operator

#python 2


class IncrementalClustering:
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractproperty
    def count(self):
        pass

    @abstractmethod
    def add_many_time_series(self, time_series_list):
        pass

    @abstractmethod
    #kwargs options
    def get_cluster_list(self, database, **kwargs):
        pass

    @abstractmethod
    def is_fitted(self, **kwargs):
        pass

    @abstractmethod
    def fit(self, **kwargs):
        pass

    #kwargs options
    @abstractmethod
    def get_number_of_clusters(self, **kwargs):
        pass


class Birch(IncrementalClustering):

    def __init__(self, threshold, cluster_distance_measure='d0', cluster_size_measure='r', n_global_clusters=50, branching_factor=50, remove_outliers=False, outlier_rate=0.1):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.cluster_size_measure = cluster_size_measure
        self.cluster_distance_measure = cluster_distance_measure
        self._data_ids = []
        self.root = BirchNode(self, True)
        self._locally_labeled_data = None
        self._globally_labeled_data = None
        self.n_global_clusters = n_global_clusters
        self._remove_outliers = remove_outliers
        self._outlier_rate = outlier_rate

    @property
    def count(self):
        cfs = self.root._clustering_features
        result = 0
        for cf in cfs:
            result += cf.count
        return result

    def add_many_time_series(self, time_series_list):
        self._locally_labeled_data = None
        self._globally_labeled_data = None
        for time_series in time_series_list:
            self._try_add_one_time_series(time_series)

    def get_cluster_list(self, **kwargs):
        global_clusters = False
        if 'mode' in kwargs and kwargs['mode'] == 'global':
            global_clusters = True
        if global_clusters:
            labeled_data = self.globally_labeled_data
            unique_labels = self.unique_global_labels
            centers = self.global_centers
        else:
            labeled_data = self.locally_labeled_data
            unique_labels = self.unique_local_labels
            centers = self.centers
        cluster_list = []
        data, labels = labeled_data.T
        labels = labels.astype(np.float32).astype(np.int32)
        for center, label in itertools.izip(centers, unique_labels):
            lc_indices = data[np.where(labels == label)[0]].tolist()
            cluster_list.append(lc_indices)
        return centers, cluster_list

    def is_fitted(self, **kwargs):
        global_clusters = False
        if 'mode' in kwargs and kwargs['mode'] == 'global':
            global_clusters = True
        if global_clusters:
            return self.has_global_labels
        else:
            return self.has_local_labels

    def fit(self, **kwargs):
        global_clusters = False
        if 'mode' in kwargs and kwargs['mode'] == 'global':
            global_clusters = True
        if global_clusters:
            self._do_global_clustering()
        else:
            self._generate_labels()

    def get_number_of_clusters(self, **kwargs):
        global_clusters = False
        if 'mode' in kwargs and kwargs['mode'] == 'global':
            global_clusters = True
        if global_clusters:
            return self.number_of_global_labels
        else:
            return self.number_of_local_labels

    @property
    def has_local_labels(self):
        return self._locally_labeled_data is not None

    @property
    def has_global_labels(self):
        return self._globally_labeled_data is not None

    @property
    def locally_labeled_data(self):
        if not self.has_local_labels:
            self._generate_labels()
        return self._locally_labeled_data

    @property
    def globally_labeled_data(self):
        if not self.has_global_labels:
            self._do_global_clustering()
        return self._globally_labeled_data

    @property
    def centers(self):
        if not self.has_local_labels:
            self._generate_labels()
        return (self._linear_sums.T / self._counts).T

    @property
    def squared_norms(self):
        if not self.has_local_labels:
            self._generate_labels()
        return self._squared_norms

    @property
    def linear_sums(self):
        if not self.has_local_labels:
            self._generate_labels()
        return self._linear_sums

    @property
    def counts(self):
        if not self.has_local_labels:
            self._generate_labels()
        return self._counts

    @property
    def global_centers(self):
        if not self.has_local_labels:
            self._do_global_clustering()
        return self._global_centers

    @property
    def unique_local_labels(self):
        if not self.has_local_labels:
            self._generate_labels()
        return list(set(self.locally_labeled_data[:,1].astype(np.float32)
                        .astype(np.int32).tolist()))

    @property
    def unique_global_labels(self):
        if not self.has_global_labels:
            self._do_global_clustering()
        return list(set(self.globally_labeled_data[:,1].astype(np.float32)
                        .astype(np.int32).tolist()))
    @property
    def number_of_local_labels(self):
        if not self.has_local_labels:
            return None
        return len(self.unique_local_labels)

    @property
    def number_of_global_labels(self):
        if not self.has_global_labels:
            return None
        return len(self.unique_global_labels)

    def _generate_labels(self):
        clusters = self.root.get_clusters()
        counts = []
        squared_norms = []
        linear_sums = []
        clusters_data_ids = []
        for cluster in clusters:
            data_ids = cluster.get_indices()
            counts.append(cluster.count)
            squared_norms.append(cluster.squared_norm)
            linear_sums.append(cluster.linear_sum)
            clusters_data_ids.append(data_ids)
        counts = np.array(counts)
        linear_sums = np.array(linear_sums)
        squared_norms = np.array(squared_norms)
        if self._remove_outliers:
            count_avg = np.mean(counts)
            count_rate = counts / count_avg
            not_outliers = np.where(count_rate > self._outlier_rate)[0]
            counts =



        self._locally_labeled_data = labels
        self._counts = np.array(counts)
        self._linear_sums = np.vstack(linear_sums)
        self._squared_norms = np.array(squared_norms)

    def _do_global_clustering(self):
        counts = self.counts
        linear_sums = self.linear_sums
        squared_norms = self.squared_norms
        n_clusters = len(counts)
        distances = []
        indices_dict = {}
        for i in range(n_clusters):
            indices_dict[i] = [i]
        for i, j in itertools.product(range(n_clusters), repeat=2):
            distance = np.inf
            if i < j:
                distance = self.d2(counts[i], linear_sums[i], squared_norms[i], counts[j], linear_sums[j], squared_norms[j])
            distances.append(distance)
        distances = np.array(distances)
        n_global_clusters = n_clusters
        while n_global_clusters > self.n_global_clusters:
            min_index = np.argmin(distances)
            min_i, min_j = min_index/n_clusters, min_index%n_clusters
            distances[min_index] = np.inf
            indices_dict[min_i] += indices_dict[min_j]
            del indices_dict[min_j]
            j_indices = []
            for k in range(n_clusters):
                if k != min_i and k != min_j:
                    if k < min_i:
                        index_i = k*n_clusters + min_i
                    else:
                        index_i = min_i*n_clusters + k
                    if k < min_j:
                        index_j = k*n_clusters + min_j
                    else:
                        index_j = min_j*n_clusters + k
                    distance_i = distances[index_i]
                    distance_j = distances[index_j]
                    count_i = counts[min_i]
                    count_j = counts[min_j]
                    new_distance = np.sqrt((count_i*distance_i**2 + count_j*distance_j**2)/(count_i + count_j))
                    distances[index_i] = new_distance
                    j_indices.append(index_j)
            distances[j_indices] = np.inf
            n_global_clusters -= 1
        indices_list = list(indices_dict.values())
        #print(indices_list)
        self._build_global_clusters(indices_list)

    def _build_global_clusters(self, indices_list):
        global_centers = []
        global_labels = []
        next_global_label = 0
        labels = self._locally_labeled_data[:, 1].astype(np.float32).astype(np.int32)
        #print labels
        for cluster_indices in indices_list:
            linear_sum = np.sum(self.linear_sums[cluster_indices], axis=0)
            count = np.sum(self.counts[cluster_indices])
            center = linear_sum / count
            global_centers.append(center)
            index_mask = []
            for index in cluster_indices:
                index_mask.append(labels == index)

            index_mask = np.vstack(index_mask)
            index_mask = np.any(index_mask, axis=0)
            data_indices = self._locally_labeled_data[index_mask, 0]
            global_cluster_labels = np.column_stack((data_indices, next_global_label*np.ones(len(data_indices))))
            global_labels.append(global_cluster_labels)
            next_global_label += 1
        self._global_centers = np.vstack(global_centers)
        self._globally_labeled_data = np.vstack(global_labels)
        #print self._globally_labeled_data

    def violates_threshold(self, count, linear_sum, squared_norm):
        return self.cluster_size(count, linear_sum, squared_norm) >= self.threshold

    def cluster_size(self, count, linear_sum, squared_norm):
        if self.cluster_size_measure == 'd':
            return Birch.diameter(count, linear_sum, squared_norm)
        else:
            return Birch.radius(count, linear_sum, squared_norm)

    def cluster_distance(self, count_1, linear_sum_1, squared_norm1, count_2, linear_sum_2, squared_norm2):
        if self.cluster_distance_measure == 'd1':
            return Birch.d1(count_1, linear_sum_1, squared_norm1, count_2, linear_sum_2, squared_norm2)
        elif self.cluster_distance_measure == 'd2':
            return Birch.d2(count_1, linear_sum_1, squared_norm1, count_2, linear_sum_2, squared_norm2)
        elif self.cluster_distance_measure == 'd3':
            return Birch.d3(count_1, linear_sum_1, squared_norm1, count_2, linear_sum_2, squared_norm2)
        else:
            return Birch.d0(count_1, linear_sum_1, squared_norm1, count_2, linear_sum_2, squared_norm2)

    def _try_add_one_time_series(self, time_series):
        reduced_vector = time_series.reduced_vector
        id_ = time_series.id
        if id_ not in self._data_ids:
            if reduced_vector is not None and len(reduced_vector):
                self._add_data_point(id_, np.array(reduced_vector))

    def _add_data_point(self, id_, data_point):
        self._globally_labeled_data = None
        self._locally_labeled_data = None
        squared_norm = np.linalg.norm(data_point)**2
        data_point_cf = data_point, squared_norm
        self._data_ids.append(id_)
        self.root.add(id_, data_point_cf)

    @staticmethod
    def to_float(count, linear_sum, squared_norm):
        return float(count), linear_sum.astype(np.float32), squared_norm.astype(np.float32)

    @staticmethod
    def radius(count, linear_sum, squared_norm):
        count, linear_sum, squared_norm = Birch.to_float(count, linear_sum, squared_norm)
        centroid = linear_sum/count
        result = np.sqrt(squared_norm/count - np.linalg.norm(centroid)**2)
        return result

    @staticmethod
    def diameter(count, linear_sum, squared_norm):
        count, linear_sum, squared_norm = Birch.to_float(count, linear_sum, squared_norm)
        return np.sqrt(2)*Birch.radius(count, linear_sum, squared_norm)

    @staticmethod
    def d0(count_1, linear_sum_1, squared_norm_1, count_2, linear_sum_2, squared_norm_2):
        count_1, linear_sum_1, squared_norm_1 = Birch.to_float(count_1, linear_sum_1, squared_norm_1)
        count_2, linear_sum_2, squared_norm_2 = Birch.to_float(count_2, linear_sum_2, squared_norm_2)
        centroid_1 = linear_sum_1/count_1
        centroid_2 = linear_sum_2/count_2
        return np.linalg.norm(centroid_1-centroid_2)**2

    @staticmethod
    def d1(count_1, linear_sum_1, squared_norm_1, count_2, linear_sum_2, squared_norm_2):
        count_1, linear_sum_1, squared_norm_1 = Birch.to_float(count_1, linear_sum_1, squared_norm_1)
        count_2, linear_sum_2, squared_norm_2 = Birch.to_float(count_2, linear_sum_2, squared_norm_2)
        centroid_1 = linear_sum_1/count_1
        centroid_2 = linear_sum_2/count_2
        return np.sum(np.abs(centroid_1-centroid_2))

    @staticmethod
    def d2(count_1, linear_sum_1, squared_norm1, count_2, linear_sum_2, squared_norm2):
        return np.sqrt(squared_norm1/count_1 + squared_norm2/count_2 - 2*np.dot(linear_sum_1, linear_sum_2)/count_1/count_2)

    @staticmethod
    def d3(count_1, linear_sum_1, squared_norm_1, count_2, linear_sum_2, squared_norm_2):
        count_1, linear_sum_1, squared_norm_1 = Birch.to_float(count_1, linear_sum_1, squared_norm_1)
        count_2, linear_sum_2, squared_norm_2 = Birch.to_float(count_2, linear_sum_2, squared_norm_2)
        return Birch.diameter(count_1+count_2,linear_sum_1+linear_sum_2,squared_norm_1+squared_norm_2)

    @staticmethod
    def d4(count_1, linear_sum_1, squared_norm1, count_2, linear_sum_2, squared_norm2):
        count = count_1 + count_2
        ss = squared_norm1 + squared_norm2
        ls = linear_sum_1 + linear_sum_2
        result_merged = count * Birch.radius(count, ls, ss)**2
        result_1 = count_1 * Birch.radius(count_1, linear_sum_1, squared_norm1)**2
        result_2 = count_2 * Birch.radius(count_2, linear_sum_2, squared_norm2)**2
        result = result_merged - result_1 - result_2
        return result

    @staticmethod
    def to_float(count, linear_sum, squared_norm):
        return float(count), linear_sum.astype(np.float32), squared_norm.astype(np.float32)

    @staticmethod
    def radius(count, linear_sum, squared_norm):
        count, linear_sum, squared_norm = Birch.to_float(count, linear_sum, squared_norm)
        centroid = linear_sum/count
        result = np.sqrt(squared_norm/count - np.linalg.norm(centroid)**2)
        return result

    @staticmethod
    def diameter(count, linear_sum, squared_norm):
        count, linear_sum, squared_norm = Birch.to_float(count, linear_sum, squared_norm)
        return np.sqrt(2)*Birch.radius(count, linear_sum, squared_norm)

    @staticmethod
    def d0(count_1, linear_sum_1, squared_norm_1, count_2, linear_sum_2, squared_norm_2):
        count_1, linear_sum_1, squared_norm_1 = Birch.to_float(count_1, linear_sum_1, squared_norm_1)
        count_2, linear_sum_2, squared_norm_2 = Birch.to_float(count_2, linear_sum_2, squared_norm_2)
        centroid_1 = linear_sum_1/count_1
        centroid_2 = linear_sum_2/count_2
        return np.linalg.norm(centroid_1-centroid_2)**2

    @staticmethod
    def d1(count_1, linear_sum_1, squared_norm_1, count_2, linear_sum_2, squared_norm_2):
        count_1, linear_sum_1, squared_norm_1 = Birch.to_float(count_1, linear_sum_1, squared_norm_1)
        count_2, linear_sum_2, squared_norm_2 = Birch.to_float(count_2, linear_sum_2, squared_norm_2)
        centroid_1 = linear_sum_1/count_1
        centroid_2 = linear_sum_2/count_2
        return np.sum(np.abs(centroid_1-centroid_2))

    @staticmethod
    def d2(count_1, linear_sum_1, squared_norm1, count_2, linear_sum_2, squared_norm2):
        return np.sqrt(squared_norm1/count_1 + squared_norm2/count_2 - 2*np.dot(linear_sum_1, linear_sum_2)/count_1/count_2)

    @staticmethod
    def d3(count_1, linear_sum_1, squared_norm_1, count_2, linear_sum_2, squared_norm_2):
        count_1, linear_sum_1, squared_norm_1 = Birch.to_float(count_1, linear_sum_1, squared_norm_1)
        count_2, linear_sum_2, squared_norm_2 = Birch.to_float(count_2, linear_sum_2, squared_norm_2)
        return Birch.diameter(count_1+count_2,linear_sum_1+linear_sum_2,squared_norm_1+squared_norm_2)

    @staticmethod
    def d4(count_1, linear_sum_1, squared_norm1, count_2, linear_sum_2, squared_norm2):
        count = count_1 + count_2
        ss = squared_norm1 + squared_norm2
        ls = linear_sum_1 + linear_sum_2
        result_merged = count * Birch.radius(count, ls, ss)**2
        result_1 = count_1 * Birch.radius(count_1, linear_sum_1, squared_norm1)**2
        result_2 = count_2 * Birch.radius(count_2, linear_sum_2, squared_norm2)**2
        result = result_merged - result_1 - result_2
        return result


class BirchNode:

    def __init__(self, birch, is_leaf = False):
        self.birch = birch
        self._clustering_features = []
        self.is_leaf = is_leaf
        self.cf_parent = None

    @property
    def size(self):
        return len(self._clustering_features)

    @property
    def has_to_split(self):
        return len(self._clustering_features) > self.birch.branching_factor

    @property
    def is_full(self):
        return len(self._clustering_features) >= self.birch.branching_factor

    @property
    def cf_sum(self):
        count_sum = 0
        linear_sum_sum = 0
        squared_norm_sum = 0
        for cf in self._clustering_features:
            count_sum += cf.count
            linear_sum_sum += cf.linear_sum
            squared_norm_sum += cf.squared_norm
        return count_sum, linear_sum_sum, squared_norm_sum

    @property
    def is_root(self):
        return self.cf_parent is None

    @property
    def node_parent(self):
        if self.cf_parent is None:
            return None
        else:
            return self.cf_parent.node

    def add(self, index, data_point_cf):
        data_point, squared_norm = data_point_cf
        if len(self._clustering_features) == 0:
            new_cf = LeafClusteringFeature(self.birch)
            self.add_clustering_feature(new_cf)
            new_cf.add(index, data_point_cf)
        else:
            distances = []
            for cf in self._clustering_features:
                distance = cf.distance(1, data_point, squared_norm)
                distances.append(distance)
            best_cf = self._clustering_features[np.argmin(distances)]
            can_be_added = best_cf.can_add(index, data_point_cf)
            if can_be_added:
                best_cf.add(index, data_point_cf)
            else:
                new_cf = LeafClusteringFeature(self.birch)
                self.add_clustering_feature(new_cf)
                new_cf.add(index, data_point_cf)
                if self.has_to_split:
                    self.split()

    def add_clustering_feature(self, cf):
        self._clustering_features.append(cf)
        cf.node = self
        if not self.is_root and not cf.is_empty:
            self.cf_parent.update(cf.count, cf.linear_sum, cf.squared_norm)

    #returns tuple with center and indices of leaf clusters
    def get_clusters(self):
        clusters = []
        if self.is_leaf:
            for cf in self._clustering_features:
                clusters.append(cf)
        else:
            for cf in self._clustering_features:
                clusters += cf.get_clusters()
        return clusters

    #only use on splits
    def replace_cf(self, old_cf, cf1, cf2):
        self._clustering_features.remove(old_cf)
        self._clustering_features.append(cf1)
        self._clustering_features.append(cf2)
        cf1.node = self
        cf2.node = self

    #only use on merges
    def merge_replace_cf(self, new_cf, cf1, cf2):
        self._clustering_features.append(new_cf)
        self._clustering_features.remove(cf1)
        self._clustering_features.remove(cf2)
        new_cf.node = self

    def merging_refinement(self, splitted_cf0, splitted_cf1):
        distances = {}
        i = 0
        cfs = self._clustering_features
        for cf1 in cfs:
            j = i + 1
            for cf2 in cfs[j:]:
                distances[(i, j)] = self.birch.cluster_distance(cf1.count, cf1.linear_sum, cf1.squared_norm, cf2.count, cf2.linear_sum, cf2.squared_norm)
                j += 1
            i += 1
        seeds_indices = min(distances.iteritems(), key=operator.itemgetter(1))[0]
        merger0 = cfs[seeds_indices[0]]
        merger1 = cfs[seeds_indices[1]]
        if merger0 is splitted_cf0 and merger1 is splitted_cf1 or merger0 is splitted_cf1 and merger1 is splitted_cf0:
            return
        new_node = BirchNode(self.birch, self.is_leaf)
        new_cf = NonLeafClusteringFeature(self.birch, new_node)
        mergers_cfs = merger0.child._clustering_features + merger1.child._clustering_features
        for cf in mergers_cfs:
            new_node.add_clustering_feature(cf)
        self.merge_replace_cf(new_cf, merger0, merger1)

    def split(self):
        last_split = True
        new_node0 = BirchNode(self.birch, self.is_leaf)
        new_node1 = BirchNode(self.birch, self.is_leaf)
        new_cf0 = NonLeafClusteringFeature(self.birch, new_node0)
        new_cf1 = NonLeafClusteringFeature(self.birch, new_node1)

        cfs = self._clustering_features
        distances = {}
        i = 0
        for cf1 in cfs:
            j = i + 1
            for cf2 in cfs[j:]:
                distances[(i, j)] = self.birch.cluster_distance(cf1.count, cf1.linear_sum, cf1.squared_norm, cf2.count, cf2.linear_sum, cf2.squared_norm)
                j += 1
            i += 1
        seeds_indices = max(distances.iteritems(), key=operator.itemgetter(1))[0]
        seed0 = cfs[seeds_indices[0]]
        seed1 = cfs[seeds_indices[1]]
        new_node0.add_clustering_feature(seed0)
        new_node1.add_clustering_feature(seed1)
        cfs.remove(seed0)
        cfs.remove(seed1)
        new_nodes = [new_node0, new_node1]
        while len(cfs) != 0:
            next_cf = cfs[0]
            dist0 = self.birch.cluster_distance(new_cf0.count, new_cf0.linear_sum, new_cf0.squared_norm, next_cf.count, next_cf.linear_sum, next_cf.squared_norm)
            dist1 = self.birch.cluster_distance(new_cf1.count, new_cf1.linear_sum, new_cf1.squared_norm, next_cf.count, next_cf.linear_sum, next_cf.squared_norm)
            closest_node, furthest_node = [new_nodes[i] for i in np.argsort([dist0, dist1])]
            if closest_node.is_full:
                furthest_node.add_clustering_feature(next_cf)
            else:
                closest_node.add_clustering_feature(next_cf)
            cfs.remove(next_cf)

        if self.is_root:
            new_root = BirchNode(self.birch, False)
            new_root.add_clustering_feature(new_cf0)
            new_root.add_clustering_feature(new_cf1)
            self.birch.root = new_root
            last_split_node = new_root
        else:
            self.node_parent.replace_cf(self.cf_parent, new_cf0, new_cf1)
            if self.node_parent.has_to_split:
                last_split = False
                self.node_parent.split()
            else:
                last_split_node = self.node_parent
        if last_split:
            last_split_node.merging_refinement(new_cf0, new_cf1)


class ClusteringFeature:

    __metaclass__ = ABCMeta

    def __init__(self, birch, count, linear_sum, squared_norm):
        self.birch = birch
        self.linear_sum = linear_sum
        self.squared_norm = squared_norm
        self.count = count
        self.node = None

    @property
    def cf_parent(self):
        if self.node is None:
            return None
        return self.node.cf_parent

    @property
    def is_empty(self):
        return self.count == 0

    @property
    def centroid(self):
        return self.linear_sum / self.count

    def update(self, count_increment, linear_sum_increment, squared_norm_increment):
        self.linear_sum += linear_sum_increment
        self.squared_norm += squared_norm_increment
        self.count += count_increment
        if self.cf_parent is not None:
            self.cf_parent.update(count_increment, linear_sum_increment, squared_norm_increment)

    def distance(self, count, linear_sum, squared_norm):
        return self.birch.cluster_distance(self.count, self.linear_sum, self.squared_norm, count, linear_sum, squared_norm)

    @abstractmethod
    def can_add(self, index, data_point_cf):
        pass

    @abstractmethod
    def add(self, index, data_point_cf):
        pass


class LeafClusteringFeature(ClusteringFeature):

    def __init__(self, birch):
        self.data_indices = []
        super(LeafClusteringFeature, self).__init__(birch, 0, 0, 0)

    def can_add(self, index, data_point_cf):
        data_point, squared_norm = data_point_cf
        new_linear_sum = self.linear_sum + data_point
        new_squared_norm = self.squared_norm + squared_norm
        new_count = self.count + 1
        if not self.birch.violates_threshold(new_count, new_linear_sum, new_squared_norm):
            return True
        return False

    def add(self, index, data_point_cf):
        data_point, squared_norm = data_point_cf
        self.update(1, data_point, squared_norm)
        self.data_indices.append(index)

    def get_indices(self):
        return self.data_indices


class NonLeafClusteringFeature(ClusteringFeature):

    def __init__(self, birch, child_node):
        count = 0
        linear_sum = 0
        squared_norm = 0
        super(NonLeafClusteringFeature, self).__init__(birch, count, linear_sum, squared_norm)
        self.set_child(child_node)

    def set_child(self, child_node):
        self.child = child_node
        child_node.cf_parent = self
        self.count, self.linear_sum, self.squared_norm = child_node.cf_sum

    def can_add(self, index, data_point):
        return True

    def get_clusters(self):
        return self.child.get_clusters()

    def add(self, index, data_point_cf):
        self.child.add(index, data_point_cf)


class IncrementalDimensionalityReduction:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def add_many_time_series(self):
        pass

    @abstractmethod
    def transform_time_series(self):
        pass

    @abstractmethod
    def add_transform_time_series(self):
        pass


class IncrementalPCA(IncrementalDimensionalityReduction):

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.cov = None
        self._W = None
        self.data_ids = []

    @staticmethod
    def standarize(X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        return np.nan_to_num((X - mean)/std)

    @staticmethod
    def cov(X):
        X_std = IncrementalPCA.standarize(X)
        n = len(X)
        return np.nan_to_num(np.matrix(X_std).T*np.matrix(X_std))/n

    @staticmethod
    def xtx(cov, std, mean, n):
        a = std.T*std
        b = mean.T*mean
        xtx_ = n*np.multiply(a,cov)
        return xtx_ + n*b

    @staticmethod
    def var_inc(d, new_row, x_mean, n):
        new_row_sq = np.matrix(np.array(new_row)**2)
        return (d + new_row_sq)/(n + 1) - np.matrix(np.array((n*x_mean + new_row)/(n+1))**2)

    @staticmethod
    def cov_stack(x2, x1_mean, x1_cov, x1_std, n1):
        xtx1 = IncrementalPCA.xtx(x1_cov, x1_std, x1_mean, n1)
        xtx2 = x2.T*x2
        n2 = len(x2)
        n = n1 + n2
        x2_mean = np.mean(x2, axis=0)
        xtx_stack = xtx1 + xtx2
        d1 = np.matrix(np.diag(xtx1))
        std_stack = np.sqrt(IncrementalPCA.var_stack(d1, x2, x1_mean, n1))
        a = std_stack.T*std_stack
        x_stack_mean = (n1*x1_mean + n2*x2_mean)/n
        b = xtx_stack - n*x_stack_mean.T*x_stack_mean
        return np.nan_to_num(np.true_divide(b,a))/n, x_stack_mean, std_stack

    @staticmethod
    def var_stack(d1, x2, x1_mean, n1):
        d2 = np.diag(x2.T*x2)
        n2 = len(x2)
        x2_mean = np.mean(x2, axis=0)
        d1 = np.float128(d1)
        x1_mean = np.float128(x1_mean)
        n1 = np.float128(n1)
        d2 = np.float128(d2)
        x2_mean = np.float128(x2_mean)
        n2 = np.float128(n2)
        return np.float32((d1 + d2)/(n1 + n2) - np.matrix(np.array((n1*x1_mean + n2*x2_mean)/(n1+n2))**2))

    @property
    def W(self):
        if self._W is None:
            self._calculate_W()
        return self._W

    def _calculate_W(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.cov)
        eigenvalues_order = np.argsort(eigenvalues)[::-1]
        #sorted_eigenvalues = eigenvalues[eigenvalues_order]
        sorted_eigenvectors = eigenvectors[:,eigenvalues_order]
        self._W = sorted_eigenvectors
        if self.n_components is not None:
            self._W = self._W[:,:self.n_components]

    def _transform_data_matrix(self, X):
        standarized_X = IncrementalPCA.standarize(X)
        return np.dot(standarized_X, self.W)

    def _add_data_matrix(self, x, ids):
        x = np.matrix(x)
        if x.shape[0] != len(ids):
            raise ValueError('Number of data points must match ids length')
        self.data_ids += ids
        self._W = None
        if self.cov is None:
            self.cov = IncrementalPCA.cov(x)
            self.mean = np.mean(x, axis= 0)
            self.std = np.std(x, axis=0)
            self.n = len(x)
        else:
            self.cov, self.mean, self.std = IncrementalPCA.cov_stack(x, self.mean, self.cov, self.std, self.n)
            # if np.isnan(np.sum(self.cov)):
            #     print(self.cov)
            n = self.n + len(x)

    def _extract_feature_matrix(self, time_series_list, only_absents):
        ids = []
        feature_vectors = []
        for time_series in time_series_list:
            id_ = time_series.id
            feature_vector = time_series.reduced_vector
            if only_absents and id_ in self.data_ids:
                continue
            if len(feature_vector) != 0:
                ids.append(id_)
                feature_vectors.append(feature_vector)
        return np.matrix(np.vstack(feature_vectors)), ids

    @staticmethod
    def _update_time_series(time_series_list, reduced_matrix):
        for reduced_vector, time_series in zip(reduced_matrix, time_series_list):
            time_series.set_reduced(np.array(reduced_vector).flatten())

    def add_many_time_series(self, time_series_list):
        feature_matrix, ids = self._extract_feature_matrix(time_series_list, True)
        self._add_data_matrix(feature_matrix, ids)
        return feature_matrix

    def transform_time_series(self, time_series_list):
        feature_matrix, ids = self._extract_feature_matrix(time_series_list, False)
        reduced_matrix = self._transform_data_matrix(feature_matrix)
        IncrementalPCA._update_time_series(time_series_list, reduced_matrix)
        return reduced_matrix

    def add_transform_time_series(self, time_series_list):
        feature_matrix, ids = self._extract_feature_matrix(time_series_list, True)
        self._add_data_matrix(feature_matrix, ids)
        reduced_matrix = self._transform_data_matrix(feature_matrix)
        IncrementalPCA._update_time_series(time_series_list, reduced_matrix)
        return feature_matrix, reduced_matrix








