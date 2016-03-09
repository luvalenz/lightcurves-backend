__author__ = 'lucas'

import json, os, glob
import pandas as pd
import numpy as np
from offline_algorithms import Birch, IncrementalPCA as IPCA
from ..data_model.time_series import MongoTimeSeriesDataBase, MachoFileDataBase
from ..data_model.clusters import ClustersMongoDataBase, Cluster
from ..data_model.serializations import SerializationMongoDatabase



class OfflineInterface(object):

    __instance = None

    def __new__(cls):
        if cls.__instance is None:
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
        serializing_db = self.get_serialization_database(serializing_db_index)
        time_series_db.setup()
        clustering_db.setup()
        serializing_db.setup()

    def defragment_clusters(self, clustering_db_index=0):
        clustering_db = self.get_clustering_database(clustering_db_index)
        clustering_db.defragment()

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
            return Model(**parameters)

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
            return Model(**parameters)

    def get_time_series_database(self, index=0):
        db_info = self.config['time_series_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = MongoTimeSeriesDataBase
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

    def get_serialization_database(self, index):
        db_info = self.config['serializing_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = SerializationMongoDatabase
        return Database(**parameters)

    def transfer_time_series(self, source_database_index, destination_database_index=0):
        source_db = self.get_time_series_database(source_database_index)
        destination_db = self.get_time_series_database(destination_database_index)
        batch_iterable = source_db.get_all(1)#TODO ELIMINATE THAT '1'
        for batch in batch_iterable:
            destination_db.add_many(batch)

    def calculate_all_features(self, database_index):
        time_series_db = self.get_time_series_database(database_index)
        batch_iterable = time_series_db.get_all()
        for batch in batch_iterable:
            updated = []
            for time_series in batch:
                if self.feature_vector is None or len(self.feature_vector) == 0:
                    time_series.calculate_features()
                    updated.append(time_series)
            time_series_db.update_many(updated)

    def reduce_all(self, serialization_db_index=0, reduction_model_index=0, time_series_db_index=0):
        time_series_db = self.get_time_series_database(time_series_db_index)
        batch_iterable = time_series_db.get_all()
        reduction_model = self.get_reduction_model(serialization_db_index, reduction_model_index)
        serialization_db = self.get_serialization_database(serialization_db_index)
        for time_series_list in batch_iterable:
            reduction_model.add_transform_time_series(time_series_list)
            time_series_db.update_many(time_series_list)
        serialization_db.store_reduction_model(reduction_model)

    def cluster_all(self, time_series_database_index=0,
                    serialization_db_index=0, clustering_model_index=0):
        time_series_db = self.get_time_series_database(time_series_database_index)
        clustering_model = self.get_clustering_model(serialization_db_index, clustering_model_index)
        serialization_db = self.get_serialization_database(serialization_db_index)
        batch_iterable = time_series_db.get_all()
        for time_series_list in batch_iterable:
            clustering_model.add_many_time_series(time_series_list)
        serialization_db.store_clustering_model(clustering_model)

    def store_all_clusters(self, serialization_db_index=0, clustering_model_index=0, clustering_db_index=0):
        clustering_model = self.get_clustering_model(serialization_db_index, clustering_model_index)
        clustering_db = self.get_clustering_database(clustering_db_index)
        clustering_db.reset_database()





