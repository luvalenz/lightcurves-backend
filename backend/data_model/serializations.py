from abc import ABCMeta, abstractmethod, abstractproperty

import pymongo
from pymongo import MongoClient
import cPickle as pickle
from bson.binary import Binary
import gridfs
import os


class SerializationDatabase(object):

    __metaclass_ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractproperty
    def has_reduction_model(self):
        pass

    @abstractproperty
    def has_clustering_model(self):
        pass

    @abstractproperty
    def reduction_model(self):
        pass

    @abstractproperty
    def clustering_model(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def store_reduction_model(self):
        pass

    @abstractmethod
    def store_clustering_model(self):
        pass


class SerializationMongoDatabase(object):

    __metaclass_ = ABCMeta

    def __init__(self, db_name='serializations', url='localhost', port=27017):
        client = MongoClient(url, port)
        self.db = client[db_name]

    @property
    def has_reduction_model(self):
        fs = gridfs.GridFS(self.db)
        filename = "reduction"
        return fs.exists({"filename": filename})

    @property
    def has_clustering_model(self):
        fs = gridfs.GridFS(self.db)
        filename = "clustering"
        return fs.exists({"filename": filename})

    @property
    def reduction_model(self):
        fs = gridfs.GridFS(self.db)
        filename = "reduction"
        binary = fs.find_one({"filename": filename}, no_cursor_timeout=True).read()
        return pickle.loads(binary)

    @property
    def clustering_model(self):
        fs = gridfs.GridFS(self.db)
        filename = "clustering"
        binary = fs.find_one({"filename": filename}, no_cursor_timeout=True).read()
        return pickle.loads(binary)

    def _get_old_clustering_model(self):
        binary = self.db['models'].find_one({'model': 'reduction'})['bin-data']
        return pickle.loads(binary)

    def setup(self):
        collection = self.db['models']
        collection.create_index([("model", pymongo.ASCENDING)], background=True, unique=True)

    def store_reduction_model(self, model):
        fs = gridfs.GridFS(self.db)
        binary_data = pickle.dumps(model, 2)
        filename = "reduction"
        for grid_out in fs.find({"filename": filename},
                                no_cursor_timeout=True):
            fs.delete(grid_out._id)
        fs.put(binary_data, filename=filename)

    def store_clustering_model(self, model):
        fs = gridfs.GridFS(self.db)
        binary_data = pickle.dumps(model, 2)
        filename = 'clustering'
        for grid_out in fs.find({"filename": filename},
                                no_cursor_timeout=True):
            fs.delete(grid_out._id)
        fs.put(binary_data, filename=filename)


class SerializationPandasDatabase(object):

    __metaclass_ = ABCMeta

    def __init__(self, name, path):
        self._name = name
        self._path = path

    @property
    def has_reduction_model(self):
        reduction_path = os.path.join(self._path, 'reduction.pkl')
        return os.path.isfile(reduction_path)

    @property
    def has_clustering_model(self):
        clustering_path = os.path.join(self._path, 'clustering.pkl')
        return os.path.isfile(clustering_path)

    @property
    def reduction_model(self):
        reduction_path = os.path.join(self._path, 'reduction.pkl')
        with open(reduction_path, "rb") as input_file:
            return pickle.load(input_file)

    @property
    def clustering_model(self):
        clustering_path = os.path.join(self._path, 'clustering.pkl')
        with open(clustering_path, "rb") as input_file:
            return pickle.load(input_file)

    def setup(self):
        pass

    def store_reduction_model(self, model):
        with open(os.path.join(self._path,  "reduction.pkl"), "wb") as input_file:
            pickle.dump(model, input_file, 2)

    def store_clustering_model(self, model):
        with open(os.path.join(self._path,  "clustering.pkl"), "wb") as input_file:
            pickle.dump(model, input_file, 2)