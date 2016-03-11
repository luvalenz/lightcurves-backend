from abc import ABCMeta, abstractmethod, abstractproperty

import pymongo
from pymongo import MongoClient
import pickle
from bson.binary import Binary

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
        return self.db['models'].count({'model': 'reduction'}) != 0

    @property
    def has_clustering_model(self):
        return self.db['models'].count({'model': 'clustering'}) != 0

    @property
    def reduction_model(self):
        binary = self.db['models'].find_one({'model': 'reduction'})['bin-data']
        return pickle.loads(binary)

    @property
    def clustering_model(self):
        binary = self.db['models'].find_one({'model': 'clustering'})['bin-data']
        return pickle.loads(binary)

    def setup(self):
        collection = self.db['models']
        collection.create_index([("model", pymongo.ASCENDING)], background=True, unique=True)

    def store_reduction_model(self, model):
        binary_data = pickle.dumps(model)
        document = {'model': 'reduction', 'bin-data': Binary(binary_data)}
        self.db['models'].replace_one({'model': 'reduction'}, document, True)

    def store_clustering_model(self, model):
        binary_data = pickle.dumps(model)
        document = {'model': 'clustering', 'bin-data': Binary(binary_data)}
        self.db['models'].replace_one({'model': 'clustering'}, document, True)