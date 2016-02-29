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
    def get_radius(self, cluster_id):
        pass

    @abstractproperty
    def get_center(self, cluster_id):
        pass

    @abstractmethod
    def get_number_of_data(self, cluster_id):
        pass

    @abstractmethod
    def get_cluster_data(self, cluster_id):
        pass

    @abstractmethod
    def get_data_ids(self, cluster_id):
        pass

    @abstractmethod
    def get_cluster(self, cluster_id):
        pass


class ClustersMongoDataBase(ClustersDataBase):
    __metaclass__ = ABCMeta

    @property
    def cluster_ids(self):
        if not self._info_loaded:
            self._load_info()
        return self._ids

    @property
    def radii(self):
        if not self._info_loaded:
            self._load_info()
        return self._radii

    @property
    def centers(self):
        if not self._info_loaded:
            self._load_info()
        return self._centers

    @property
    def counts(self):
        if not self._info_loaded:
            self._load_info()
        return self._counts

    def __init__(self, url, port, db_name):
        client = MongoClient(url, port)
        self.db = client[db_name]
        self._info_loaded = False

    def store_clusters(self, clusters_list):
        if len(self.db.collection_names()) != 0:
            return
        for i, cluster in zip(range(len(clusters_list)), clusters_list):
            self._store_cluster(i, cluster)
        info_collection = self.db['info']
        info_collection.create_index([("id", pymongo.ASCENDING)])

    def _store_cluster(self, index, cluster):
        document_list = cluster.to_list_of_dicts()
        cluster_collection = self.db[str(index)]
        cluster_collection.insert_many(document_list)
        cluster_collection.create_index([("distance", pymongo.ASCENDING)])
        info_collection = self.db['info']
        info = cluster.get_info()
        info['id'] = str(index)
        info_collection.insert_one(cluster.get_info())

    def _load_info(self):
        self._info_loaded = True
        info_cursor = self.db['info'].find().sort('id', pymongo.ASCENDING)
        ids = []
        radii = []
        counts = []
        centers = []
        for cluster_info in info_cursor:
            ids.append(cluster_info['id'])
            radii.append(cluster_info['radius'])
            counts.append(cluster_info['count'])
            centers.append(cluster_info['center'])
        self._ids = ids
        self._radii = np.array(radii)
        self._counts = np.array(counts)
        self._centers = np.array(centers)

    def _get_cluster_info(self, cluster_id):
        info_collection = self.db['info']
        return info_collection.find_one({'id': str(cluster_id)})

    def get_radius(self, cluster_id):
        return self._get_cluster_info(cluster_id)['radius']

    def get_center(self, cluster_id):
        return self._get_cluster_info(cluster_id)['center']

    def get_count(self, cluster_id):
        return self._get_cluster_info(cluster_id)['count']

    def get_cluster_data(self, cluster_id):
        data_points = list(self._get_cluster_data_points_cursor(cluster_id))
        cluster = Cluster.from_list_of_dicts(data_points)
        return cluster.data_points

    def get_data_ids(self, cluster_id):
        cursor = self._get_cluster_data_points_cursor(cluster_id)
        ids = []
        for element in cursor:
            ids.append(element['id'])
        return ids

    def get_cluster(self, cluster_id):
        center = self.db['info'].find_one({'id': cluster_id})['center']
        data_points = list(self._get_cluster_data_points_cursor(cluster_id))
        return Cluster.from_list_of_dicts(data_points, cluster_id, center)

    def _get_cluster_data_points_cursor(self, cluster_id):
        return self.db[str(cluster_id)].find().sort('id', pymongo.ASCENDING)


class Cluster:

    @property
    def id(self):
        return self._id

    @property
    def radius(self):
        return self._distances[-1]

    @property
    def count(self):
        return len(self._data_point_ids)

    @property
    def data_points(self):
        return self._data_points

    @property
    def data_point_ids(self):
        return self._data_point_ids

    @property
    def center(self):
        return self._center

    def __init__(self, id_, data_ids, distances, data_points, center):
        order = np.argsort[distances]
        self._center = center
        self._data_points = np.array(data_points[order])
        self._distances = np.array(np.array(distances[order]))
        self._data_point_ids = list(data_ids[order])
        self._id = id_

    @staticmethod
    def from_pandas_data_frame(dataframe, center):
        data_points = dataframe.ix[:, dataframe.columns != 'distance'].values
        distances = dataframe['distances'].values
        data_indices = dataframe.index.values
        return Cluster(data_indices, distances, data_points, center)

    @staticmethod
    def from_list_of_dicts(data_point_list, id_=None, center=None):
        data_ids= []
        values = []
        distances = []
        for data_point_dictionary in data_point_list:
            data_ids.append(data_point_dictionary['id'])
            values.append(data_point_dictionary['values'])
            distances.append(data_point_dictionary['distance'])
        return Cluster(id_, data_ids, distances, values, center)


    def get_data_points_collection(self):
        list_of_dicts = []
        for data_point, distance in zip(self._data_points, self._distances):
            element = {'id': self._id, 'values': data_point.to_list(), 'distance': distance}
            list_of_dicts.append(element)
        return list_of_dicts

    def get_info(self):
        return {'id': self.id, 'radius': self.radius, 'count': self.count, 'center': self.center}




