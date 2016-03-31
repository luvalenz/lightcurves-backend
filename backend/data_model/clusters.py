__author__ = 'lucas'

from abc import ABCMeta, abstractmethod, abstractproperty
from pymongo import MongoClient
import numpy as np
import pymongo
from scipy.spatial.distance import cdist
import itertools
import gridfs
import cPickle as pickle

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

    @abstractproperty
    def data_points_count(self):
        pass

    @abstractproperty
    def clusters_count(self):
        pass

    @abstractproperty
    def metadata(self):
        pass

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def reset_database(self):
        pass

    @abstractmethod
    def store_cluster(self):
        pass

    @abstractproperty
    def get_radius(self, cluster_id):
        pass

    @abstractproperty
    def get_center(self, cluster_id):
        pass

    @abstractmethod
    def get_count(self, cluster_id):
        pass

    @abstractmethod
    def get_data_ids(self, cluster_id):
        pass

    @abstractmethod
    def get_cluster(self, cluster_id):
        pass

    @abstractmethod
    def defragment(self):
        pass

    @abstractmethod
    def get_all(self):
        pass


class ClustersMongoDataBase(ClustersDataBase):

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

    @property
    def data_points_count(self):
        if not self._data_points_count:
            self._load_info()
        return self._data_points_count

    @property
    def clusters_count(self):
        if not self._clusters_count:
            self._load_info()
        return self._clusters_count

    @property
    def metadata(self):
        return self._database['metadata'].find_one()

    def __init__(self, db_name='clusters', url='localhost', port=27017):
        self.client = MongoClient(url, port)
        self.db_name = db_name
        self._database = self.client[db_name]
        self._info_loaded = False

    def setup(self):
        pass

    def reset_database(self, metadata):
        self.client.drop_database(self.db_name)
        self._database = self.client[self.db_name]
        self._info_loaded = False
        info_collection = self._database['info']
        clusters_collection = self._database['clusters']
        metadata_collection = self._database['metadata']
        info_collection.create_index([("id", pymongo.ASCENDING)], background=True, unique=True)
        clusters_collection.create_index([("id", pymongo.ASCENDING)], background=True, unique=True)
        metadata_collection.insert_one(metadata)

    def store_cluster(self, index, cluster):
        data_points = cluster.to_list_of_dicts()
        clusters_collection = self._database['clusters']
        clusters_document = {'id': index, 'data': data_points}
        try:
            clusters_collection.insert_one(clusters_document)
        except pymongo.errors.DocumentTooLarge:
            fs = gridfs.GridFS(self._database)
            binary_data = pickle.dumps(clusters_document, 2)
            fs.put(binary_data, filename=str(index))
        info_collection = self._database['info']
        info = cluster.get_info()
        info['id'] = index
        info_collection.insert_one(info)

    def _load_info(self):
        self._info_loaded = True
        info_cursor = self._database['info'].find().sort('id', pymongo.ASCENDING)
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
        self._data_points_count = np.sum(self._counts)
        self._clusters_count = len(ids)

    def _get_cluster_info(self, cluster_id):
        info_collection = self._database['info']
        return info_collection.find_one({'id': int(cluster_id)})

    def get_radius(self, cluster_id):
        return self._get_cluster_info(cluster_id)['radius']

    def get_center(self, cluster_id):
        return self._get_cluster_info(cluster_id)['center']

    def get_count(self, cluster_id):
        return self._get_cluster_info(cluster_id)['count']

    def get_data_ids(self, cluster_id):
        cluster = self._database['clusters'].find(cluster_id)
        ids = []
        for element in cluster:
            ids.append(element['id'])
        return ids

    def get_cluster(self, cluster_id):
        center = self._database['info'].find_one({'id': cluster_id})['center']
        document = self._database['clusters'].find_one({'id': cluster_id})
        if document is None:
            fs = gridfs.GridFS(self._database)
            binary = fs.find_one({"filename": str(cluster_id)}, no_cursor_timeout=True).read()
            document = pickle.loads(binary)
        data_points = document['data']
        return Cluster.from_list_of_dicts(data_points, center, cluster_id, False)

    def defragment(self):
        collection_names = self._database.collection_names()
        collection_names.remove('system.indexes')
        results = {}
        for collection_name in collection_names:
            results[collection_name] = self._database.command('compact', collection_name)
        return results

    def get_all(self):
        return MongoClustersIterator(self)


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

    def __init__(self, data_ids, data_points, center, id_=None, distances=None, sort=True):
        if sort:
            if distances is None:
                distances = cdist(np.matrix(center), np.matrix(data_points))[0]
            order = np.argsort(distances)
            data_points = np.array(data_points)[order]
            distances = np.array(distances[order])
            data_ids = list(np.array(data_ids)[order])
        self._center = center
        self._id = id_
        self._data_point_ids = np.array(data_ids)
        self._data_points = np.array(data_points)
        self._distances = np.array(distances)

    @staticmethod
    def from_pandas_data_frame(dataframe, center):
        data_points = dataframe.ix[:, dataframe.columns != 'distance'].values
        data_indices = dataframe.index.values
        distances = None
        if 'distances' in dataframe:
            distances = dataframe['distances'].values
        return Cluster(data_indices, data_points, center, distances)

    @staticmethod
    def from_list_of_dicts(data_point_list, center=None, id_=None, sort=False):
        data_ids= []
        values = []
        distances = []
        for data_point_dictionary in data_point_list:
            data_ids.append(data_point_dictionary['id'])
            values.append(data_point_dictionary['values'])
            distances.append(data_point_dictionary['distance'])
        return Cluster(data_ids, values, center, id_, distances, sort)

    @staticmethod
    def from_time_series_sequence(time_series_sequence, center, id_):
        ids = []
        values = []
        for time_series in time_series_sequence:
            reduced_vector = time_series.reduced_vector
            if reduced_vector is not None:
                values.append(time_series.reduced_vector)
                ids.append(time_series.id)
        values = np.vstack(values)
        return Cluster(ids, values, center, id_)

    def to_list_of_dicts(self):
        list_of_dicts = []
        for id_, values, distance in itertools.izip(self._data_point_ids, self._data_points, self._distances):
            list_of_dicts.append({'id': id_, 'values': list(values), 'distance': distance})
        return list_of_dicts

    def get_data_points_collection(self):
        list_of_dicts = []
        for data_point, distance in zip(self._data_points, self._distances):
            element = {'id': self._id, 'values': data_point.to_list(), 'distance': distance}
            list_of_dicts.append(element)
        return list_of_dicts

    def get_ring_of_data(self, width):
        ring_indices = np.where(self._distances >= self.radius - width)[0]
        return self.data_points[ring_indices], self.data_point_ids[ring_indices]

    def get_info(self):
        return {'id': self.id, 'radius': self.radius, 'count': self.count, 'center': list(self.center)}


class ClustersIterator(object):

    def __init__(self, time_series_db, clusters, centers, batch=False, batch_size=10):
        self._batch = batch
        self._batch_size = batch_size
        self._clusters = clusters
        self._centers = centers
        self._time_series_db = time_series_db
        self._current_cluster_index = 0

    def __len__(self):
        return len(self._centers)

    def __iter__(self):
        return self

    def next_unit(self):
        if self._current_cluster_index >= len(self):
            raise StopIteration
        data_ids = self._clusters[self._current_cluster_index]
        center = self._centers[self._current_cluster_index]
        time_series_iterator = self._time_series_db.get_many(data_ids, None, False)
        cluster_obj = Cluster.from_time_series_sequence(time_series_iterator, center, self._current_cluster_index)
        self._current_cluster_index += 1
        return cluster_obj

    def next_batch(self):
        clusters_batch = []
        i = 0
        while True:
            try:
                cluster = self.next_unit()
                clusters_batch.append(cluster)
            except StopIteration:
                break
            if i == self._batch_size - 1:
                break
            i += 1
        if len(clusters_batch) == 0:
            raise StopIteration
        return clusters_batch

    def next(self):
        if self._batch:
            return self.next_batch()
        else:
            return self.next_unit()

    def rewind(self):
        self._current_cluster_index = 0


class DatabaseClustersIterator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, clusters_db):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def rewind(self):
        pass


class MongoClustersIterator(object):
    __metaclass__ = ABCMeta

    def __init__(self, clusters_db):
        self._database = clusters_db
        self._ids = self._database.cluster_ids
        self._current_index = 0

    def __len__(self):
        return len(self._ids)

    def __iter__(self):
        return self

    def next(self):
        if self._current_index >= len(self):
            raise StopIteration
        id_ = self._ids[self._current_index]
        cluster = self._database.get_cluster(id_)
        self._current_index += 1
        return cluster

    def rewind(self):
        self._current_index = 0