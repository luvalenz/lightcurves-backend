__author__ = 'lucas'

import json, os, glob
import pandas as pd
import numpy as np
from offline_algorithms import Birch, IncrementalPCA as IPCA
from ..data_model.time_series import TimeSeriesMongoDataBase, MachoFileDataBase
from ..data_model.clusters import ClustersMongoDataBase
from ..data_model.serializing import SerializingMongoDatabase


class OfflineInterface(object):

    __instance = None


    def __new__(cls):
        if cls.__instance == None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    @staticmethod
    def get_config(self):
        #print(glob.glob('*'))
        with open('config.json') as data_file:
            data = json.load(data_file)
            return data

    def __init__(self):
        #read config file
        self.config = self.get_config()

    def setup(self, time_series_db_index=0, clustering_db_index=0, serializing_db_index=0):
        time_series_db = self.get_time_series_databse(time_series_db_index)
        clustering_db = self.get_clustering_database(clustering_db_index)
        serializing_db = self.get_serializing_database(serializing_db_index)
        time_series_db.setup()
        clustering_db.setup()
        serializing_db.setup()

    #TODO
    def defragment(self):
        pass

    def get_reduction_model(self, serializing_db_index=0, model_index=0):
        serializing_db = self.get_serializing_database(serializing_db_index)
        if serializing_db.has_reduction_model:
            return serializing_db.reduction_model
        else:
            model_info = self.config['reduction_algorithms'][model_index]
            model_type = model_info['type']
            parameters = model_info['parameters']
            if model_type == 'ipca':
                Model = IPCA
            return Model(**parameters)

    def get_clustering_model(self, serializing_db_index=0, model_index=0):
        serializing_db = self.get_serializing_database(serializing_db_index)
        if serializing_db.has_clustering_model:
            return serializing_db.clustering_model
        else:
            model_info = self.config['clustering_algorithms'][model_index]
            model_type = model_info['type']
            parameters = model_info['parameters']
            if model_type == 'birch':
                Model = Birch
            return Model(**parameters)

    def get_time_series_database(self, index=0):
        db_info = self.config['time_series_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = TimeSeriesMongoDataBase
        elif model_type == 'macho':
            Database = MachoFileDataBase
        return Database(**parameters)

    def get_clustering_database(self, index=0):
        db_info = self.config['clustering_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = ClustersMongoDataBase
        return Database(**parameters)

    def get_serializing_database(self, index):
        db_info = self.config['serializing_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = SerializingMongoDatabase
        return Database(**parameters)

    def calculate_features_all(self, serializing_db_index=0, reduction_model_index=0, time_series_db_index=0):
        pass

    def reduce_all(self, serializing_db_index=0, reduction_model_index=0, time_series_db_index=0):
        reduction_model = self.get_reduction_model(serializing_db_index, reduction_model_index)
        time_series_db = self.get_time_series_database(time_series_db_index)
        reduction_model.add_
        #update reduced model
        #recalculate ALL (important)

    def cluster_all(self, time_series_database_index, cluster_database_index=0):
        pass

    def copy_database_data(self, source_database_index, destination_database_index):
        pass

