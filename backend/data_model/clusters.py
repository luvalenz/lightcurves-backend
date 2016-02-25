__author__ = 'lucas'

from abc import ABCMeta, abstractmethod, abstractproperty
from pymongo import MongoClient
import numpy as np
import pymongo


class ClustersDataBase:
    __metaclass__ = ABCMeta

    @abstractproperty
    def radii(self):
        pass

    @abstractproperty
    def centers(self):
        pass

    @abstractproperty
    def counts(self):
        pass

    @abstractmethod
    def __init__(self):
        pass

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
        if not self._is_loaded:
            self._load()
        return self._radii

    @property
    def centers(self):
        if not self._is_loaded:
            self._load()
        return self._centers

    @property
    def counts(self):
        if not self._is_loaded:
            self._load()
        return self._counts

    def __init__(self, url, port, db_name):
        client = MongoClient(url, port)
        self.db = client[db_name]
        self._is_loaded = False

    def store_clusters(self, clusters_list):
        if len(self.db.collection_names()) != 0:
            return
        for i, cluster in zip(range(len(clusters_list)), clusters_list):
            self._store_cluster(i, cluster)

    def _store_cluster(self, index, cluster):
        document_list = cluster.to_list_of_dicts()
        collection = self.db[str(index)]
        collection.insert_many(document_list)
        collection.create_index([("distance", pymongo.ASCENDING)])
        info_collection = self.db['info']
        info_dict
        info_collection.append

    def _load(self):
        self._is_loaded = True

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
        return self._distances[-1]

    @property
    def count(self):
        return len(self._indices)

    @property
    def center(self):
        return self._center

    def __init__(self, data_indices, distances, data_points, center):
        order = np.argsort[distances]
        self._center = center
        self._data_points = data_points[order]
        self._distances = distances[order]
        self._indices = data_indices[order]

    @staticmethod
    def from_pandas_data_frame(dataframe, center):
        data_points = dataframe.ix[:, dataframe.columns != 'distance'].values
        distances = dataframe['distances'].values
        data_indices = dataframe.index.values
        return Cluster(data_indices, distances, data_points, center)

    def get_data_points_collection(self):
        list_of_dicts = []
        for data_point, distance in zip(self._data_points, self._distances):
            element = {'values': data_point.to_list(), 'distance': distance}
            list_of_dicts.append(element)
        return list_of_dicts

    def get_info_document(self):
        return {'radius': self.radius, 'count': self.count, 'center': self.center}




