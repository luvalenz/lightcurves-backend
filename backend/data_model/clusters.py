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
    def __init__(self, time_series_db):
        self.time_series_db = time_series_db

    @abstractmethod
    def store_clusters(self):
        pass

    @abstractproperty
    def get_radius(self, cluster_index):
        pass

    @abstractproperty
    def centers(self, cluster_index):
        pass

    @abstractmethod
    def numbers_of_data(self, cluster_index):
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


class ClustersFileDataBase:
    pass


class ClustersMongoDataBase:
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



