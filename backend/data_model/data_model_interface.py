__author__ = 'lucas'

import os, json
from backend.offline.offline_algorithms import Birch, IncrementalPCA as IPCA, ScikitIncrementalPCAWrapper as ScikitIPCA
from backend.data_model.time_series import MongoTimeSeriesDataBase, MachoFileDataBase, PandasTimeSeriesDataBase
from backend.data_model.clusters import ClustersMongoDataBase, Cluster
from backend.data_model.serializations import SerializationMongoDatabase, SerializationPandasDatabase


def load_config(config_path):
     with open(config_path) as data_file:
        return json.load(data_file)


class DataModelInterface(object):

    def __init__(self, config):
        self._config = config

    def get_reduction_model(self, serialization_db_index=0, model_index=0):
        serialization_db = self.get_serialization_database(serialization_db_index)

        if serialization_db.has_reduction_model:
            return serialization_db.reduction_model
        else:
            model_info = self._config['reduction_algorithms'][model_index]
            model_type = model_info['type']
            parameters = model_info['parameters']
            if model_type == 'ipca':
                Model = IPCA
            elif model_type == 'scikit_ipca':
                Model = ScikitIPCA
            return Model(*parameters)

    def get_clustering_model(self, serialization_db_index=0, model_index=0):
        serialization_db = self.get_serialization_database(serialization_db_index)
        if serialization_db.has_clustering_model:
            return serialization_db.clustering_model
        else:
            model_info = self._config['clustering_algorithms'][model_index]
            model_type = model_info['type']
            parameters = model_info['parameters']
            if model_type == 'birch':
                Model = Birch
            return Model(*parameters)

    def get_time_series_database(self, index=0):
        db_info = self._config['time_series_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = MongoTimeSeriesDataBase
        elif model_type == 'macho':
            Database = MachoFileDataBase
        elif model_type == 'pandas':
            Database = PandasTimeSeriesDataBase
        if Database is PandasTimeSeriesDataBase:
            return Database.from_pickle(*parameters)
        else:
            return Database(*parameters)

    def get_clustering_database(self, index=0):
        db_info = self._config['clustering_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = ClustersMongoDataBase
        return Database(*parameters)

    def get_serialization_database(self, index=0):
        db_info = self._config['serialization_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = SerializationMongoDatabase
        elif model_type == 'pandas':
            Database = SerializationPandasDatabase
        return Database(*parameters)
