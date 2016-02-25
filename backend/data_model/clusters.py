__author__ = 'lucas'

from abc import ABCMeta, abstractmethod, abstractproperty


class ClustersDataBase:
    __metaclass__ = ABCMeta

    @abstractproperty
    def radii(self):
        pass

    @abstractproperty
    def centers(self):
        pass

    @abstractmethod
    def numbers_of_data(self):
        pass

    @abstractmethod
    def __init__(self):

    @abstractmethod
    def store_clusters(self):
        pass

    @abstractproperty
    def get_radius(self, cluster_index):
        pass

    @abstractproperty
    def get_center(self, cluster_index):
        pass

    @abstractmethod
    def get_number_of_data(self, cluster_index):
        pass

    @abstractmethod
    def get_data(self, cluster_index):
        pass

    @abstractmethod
    def get_data_indices(self, cluster_index):
        pass

    @abstractmethod
    def get_cluster(self, cluster_index):
        pass


class ClustersMongoDataBase:
    __metaclass__ = ABCMeta

    @property
    def radii(self):
        pass

    @property
    def centers(self):
        pass

    def numbers_of_data(self):
        pass

    def __init__(self):

    def store_clusters(self, clusters_list):
        if len(self.db.collection_names()) != 0:
            return


    def get_radius(self, cluster_index):
        pass

    def get_center(self, cluster_index):
        pass

    def get_number_of_data(self, cluster_index):
        pass

    def get_data(self, cluster_index):
        pass

    def get_data_indices(self, cluster_index):
        pass

    @abstractmethod
    def get_cluster(self, cluster_index):
        pass



class Cluster:

    @property
    def radius(self):
        pass

    @property
    def n_data(self):
        pass

    @property
    def center(self):
        pass

    def __init__(self, data_indices, distances, data_points, center):
        pass

    @staticmethod
    def from_pandas_data_frame(data_frame, center):
        pass

    def to_dict(self):
        pass



