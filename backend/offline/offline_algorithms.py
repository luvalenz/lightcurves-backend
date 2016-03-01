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

    @abstractproperty
    def centers(self):
        pass

    @abstractproperty
    def labels(self):
        pass

    @abstractproperty
    def unique_labels(self):
        pass

    @abstractmethod
    def generate_labels(self):
        pass

    @abstractmethod
    def generate_labels(self):
        pass

    @abstractmethod
    def to_database(self, database):
        pass

    @abstractmethod
    def add_time_series(self):
        pass


class Birch(IncrementalClustering):

    def __init__(self, threshold, cluster_distance_measure='d0', cluster_size_measure='r', n_global_clusters=50, branching_factor=50):
        self.branching_factor = branching_factor
        self.threshold = threshold
        self.cluster_size_measure = cluster_size_measure
        self.cluster_distance_measure = cluster_distance_measure
        self.root = BirchNode(self, True)
        self._labels = None
        self._global_labels = None
        self.n_global_clusters = n_global_clusters

    @property
    def has_labels(self):
        return self._labels is not None

    @property
    def has_global_labels(self):
        return self._global_labels is not None

    @property
    def count(self):
        cfs = self.root._clustering_features
        result = 0
        for cf in cfs:
            result += cf.count
        return result

    @property
    def labels(self):
        if not self.has_labels:
            self.generate_labels()
        return self._labels

    @property
    def global_labels(self):
        if not self.has_global_labels:
            self.do_global_clustering()
        return self._global_labels

    @property
    def centers(self):
        if not self.has_labels:
            self.generate_labels()
        return (self._linear_sums.T / self._counts).T

    @property
    def squared_norms(self):
        if not self.has_labels:
            self.generate_labels()
        return self._squared_norms

    @property
    def linear_sums(self):
        if not self.has_labels:
            self.generate_labels()
        return self._linear_sums

    @property
    def counts(self):
        if not self.has_labels:
            self.generate_labels()
        return self._counts

    @property
    def global_labels(self):
        if not self.has_global_labels:
            self.do_global_clustering()
        return self._global_labels

    @property
    def global_centers(self):
        if not self.has_labels:
            self.do_global_clustering()
        return self._global_centers

    @property
    def unique_labels(self):
        if not self.has_labels:
            self.generate_labels()
        return list(set(self.labels[:,1].tolist()))

    @property
    def unique_global_labels(self):
        if not self.has_global_labels:
            self.do_global_clustering()
        return list(set(self.global_labels[:,1].tolist()))

    @property
    def number_of_labels(self):
        if not self.has_labels:
            self.generate_labels()
        return len(self.unique_labels)

    @property
    def number_of_global_labels(self):
        if not self.has_global_labels:
            self.do_global_clustering()
        return len(self.unique_global_labels)

    def to_files(self, name, path):
        full_path = os.path.join(path, name)
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except OSError:
                print("Directory already exists")
        centers_df = pd.DataFrame(self.centers)
        centers_df.to_csv(os.path.join(full_path, 'centers.csv'))
        radii = []
        sizes = []
        for center, label in zip(self.centers, self.unique_labels):
            lc_indices = self.labels[np.where(self.labels[:,1] == label)[0]][:,0]
            if self.data_in_memory:
                data_points = self.data.loc[lc_indices]
                distances = dist.cdist(np.matrix(center), np.matrix(data_points.values))[0]
                data_points['distances'] = pd.Series(distances, index=data_points.index)
                sorted_data_points = data_points.iloc[np.argsort(distances)]
                sorted_data_points.to_csv(os.path.join(full_path, 'class{0}.csv'.format(int(float(label)))))
                radii.append(sorted_data_points['distances'][-1])
                sizes.append(data_points.shape[0])
        pd.DataFrame(radii).to_csv(os.path.join(full_path, 'radii.csv'))
        pd.DataFrame(sizes).to_csv(os.path.join(full_path, 'sizes.csv'))

    def to_pickle(self, name, path):
        output = open(os.path.join(path, '{0}_birch.pkl'.format(name)), 'wb')
        pickle.dump(self, output)
        output.close()

    @staticmethod
    def from_pickle(path):
        pkl_file = open(path, 'rb')
        return pickle.load(pkl_file)

    def generate_labels(self):
        clusters = self.root.get_clusters()
        labels = np.empty((0,2))
        next_label = 0
        counts = []
        squared_norms = []
        linear_sums = []
        for cluster in clusters:
            indices = cluster.get_indices()
            counts.append(cluster.count)
            squared_norms.append(cluster.squared_norm)
            linear_sums.append(cluster.linear_sum)
            cluster_labels = np.column_stack((indices, next_label*np.ones(len(indices))))
            labels = np.vstack((labels, cluster_labels))
            next_label += 1
        self._labels = labels
        self._counts = np.array(counts)
        self._linear_sums = np.vstack(linear_sums)
        self.squared_norms = np.array(squared_norms)

    def do_global_clustering(self):
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
        self._build_global_clusters(indices_list)

    def _build_global_clusters(self, indices_list):
        global_centers = []
        global_labels = []
        next_global_label = 0
        for cluster_indices in indices_list:
            linear_sum = np.sum(self.linear_sums[cluster_indices], axis=0)
            count = np.sum(self.counts[cluster_indices])
            center = linear_sum / count
            global_centers.append(center)
            index_mask = []
            for index in cluster_indices:
                index_mask.append(self.labels[:,1] == index)
            index_mask = np.vstack(index_mask)
            index_mask = np.any(index_mask, axis=0)
            data_indices = self.labels[index_mask, 0]
            global_cluster_labels = np.column_stack((data_indices, next_global_label*np.ones(len(data_indices))))
            global_labels.append(global_cluster_labels)
            next_global_label += 1
        self._global_centers = np.vstack(global_centers)
        self._global_labels = np.vstack(global_labels)

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

    def add_pandas_data_frame(self, data_frame):
        self._labels = None
        self._global_labels = None
        indices = data_frame.index.values
        data_points = data_frame.values
        for index, data_point in itertools.izip(indices, data_points):
            self.add_data_point(index, data_point)
        if self.data_in_memory:
            if self.data is None:
                self.data = data_frame
            else:
                self.data = pd.concat([self.data, data_frame])

    def add_data_point(self, index, data_point):
        squared_norm = np.linalg.norm(data_point)**2
        data_point_cf = data_point, squared_norm
        self.root.add(index, data_point_cf)

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
        global number
        self.number = number
        number += 1

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

    #TODO
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
        global number
        self.number = number
        number += 1


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
    def add_transform_time_series(self):
        pass

    @abstractmethod
    def update(self, database):
        pass


class IncrementalPCA(IncrementalDimensionalityReduction):

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.cov = None
        self._W = None

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

    def add_data_matrix(self, X):
        X = np.matrix(X)
        self._W = None
        if self.cov is None:
            self.cov = IncrementalPCA.cov(X)
            self.mean = np.mean(X, axis= 0)
            self.std = np.std(X, axis=0)
            self.n = len(X)
        else:
            self.cov, self.mean, self.std = IncrementalPCA.cov_stack(X, self.mean, self.cov, self.std, self.n)
            if np.isnan(np.sum(self.cov)):
                print(self.cov)
            n = self.n + len(X)

    def add_transform_data_matrix(self, X):
        self.add(X)
        return self._transform_data_matrix(X)

    def add_transform_time_series(self, time_series_list):
        feature_matrix = []
        for time_series in time_series_list:
            feature_matrix.append(time_series.feature_vector)
        feature_matrix = np.matrix(feature_matrix)
        reduced_matrix = self.add_transform_data_matrix(feature_matrix)
        for reduced_vector, time_series in zip(reduced_matrix, time_series_list):
            time_series.set_reduced(np.array(reduced_vector).flatten())

    def update(self, database):
        database.update_reduction_model(self)







