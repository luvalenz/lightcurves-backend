__author__ = 'lucas'

import os, json
from backend.offline.offline_algorithms import Birch, IncrementalPCA as IPCA
from backend.data_model.time_series import MongoTimeSeriesDataBase, MachoFileDataBase
from backend.data_model.clusters import ClustersMongoDataBase, Cluster
from backend.data_model.serializations import SerializationMongoDatabase


class DataModelInterface(object):

    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
            cls.__instance.set_config()
        return cls.__instance

    def set_config(self):
        config_dir = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(config_dir, 'config.json')
        with open(config_path) as data_file:
            data = json.load(data_file)
            self.config = data


    def get_reduction_model(self, serializing_db_index=0, model_index=0):
        serializing_db = self.get_serialization_database(serializing_db_index)

        if serializing_db.has_reduction_model:
            return serializing_db.reduction_model
        else:
            model_info = self.config['reduction_algorithms'][model_index]
            model_type = model_info['type']
            parameters = model_info['parameters']
            if model_type == 'ipca':
                Model = IPCA
            return Model(*parameters)

    def get_clustering_model(self, serializing_db_index=0, model_index=0):
        serializing_db = self.get_serialization_database(serializing_db_index)
        if serializing_db.has_clustering_model:
            return serializing_db.clustering_model
        else:
            model_info = self.config['clustering_algorithms'][model_index]
            model_type = model_info['type']
            parameters = model_info['parameters']
            if model_type == 'birch':
                Model = Birch
            return Model(*parameters)

    def get_time_series_database(self, index=0):
        db_info = self.config['time_series_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = MongoTimeSeriesDataBase
        elif model_type == 'macho':
            Database = MachoFileDataBase
        return Database(*parameters)

    def get_clustering_database(self, index=0):
        db_info = self.config['clustering_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = ClustersMongoDataBase
        return Database(*parameters)

    def get_serialization_database(self, index=0):
        db_info = self.config['serializing_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = SerializationMongoDatabase
        return Database(*parameters)
