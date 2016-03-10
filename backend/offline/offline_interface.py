__author__ = 'lucas'

import json, glob
import itertools
from backend.offline.offline_algorithms import Birch, IncrementalPCA as IPCA
from backend.data_model.time_series import MongoTimeSeriesDataBase, MachoFileDataBase
from backend.data_model.clusters import ClustersMongoDataBase, Cluster
from backend.data_model.serializations import SerializationMongoDatabase


class OfflineInterface(object):

    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
            cls.__instance.set_config()
        return cls.__instance

    def set_config(self):
        #print(glob.glob('*'))
        with open('../config.json') as data_file:
            data = json.load(data_file)
        self.config = data

    def __init__(self):
        pass

    #bust be called after adding elements but before the reduction
    def setup(self, time_series_db_index=0, clustering_db_index=0, serializing_db_index=0):
        time_series_db = self.get_time_series_database(time_series_db_index)
        clustering_db = self.get_clustering_database(clustering_db_index)
        serializing_db = self.get_serialization_database(serializing_db_index)
        time_series_db.setup()
        clustering_db.setup()
        serializing_db.setup()

    def transfer_time_series(self, catalog_name, source_database_index, destination_database_index=0):
        source_db = self.get_time_series_database(source_database_index)
        destination_db = self.get_time_series_database(destination_database_index)
        batch_iterable = source_db.get_all(1)#TODO ELIMINATE THAT '1'
        for batch in batch_iterable:
            if len(batch) != 0:
                print ('adding {0} elements to destination...'.format(len(batch)))
                destination_db.add_many(catalog_name, batch)

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

    def get_serialization_database(self, index):
        db_info = self.config['serializing_databases'][index]
        model_type = db_info['type']
        parameters = db_info['parameters']
        if model_type == 'mongodb':
            Database = SerializationMongoDatabase
        return Database(*parameters)

    def calculate_missing_features(self, database_index=0, batch_size=None):
        time_series_db = self.get_time_series_database(database_index)
        batch_iterable = time_series_db.find_many(None, {"features":{"$in": [{}, None]}}, True, batch_size)
        for batch in batch_iterable:
            updated = []
            for time_series in batch:
                if time_series.feature_vector is None or len(time_series.feature_vector) == 0:
                    print("Calculating features for time series {0}".format(time_series.id)),
                    time_series.calculate_features()
                    updated.append(time_series)
                    print("DONE")
            if len(updated) != 0:
                print("Updating values to database..."),
                time_series_db.update_many(updated)
                print("DONE")

    def recalculate_all_features(self, database_index=0, batch_size=None):
        time_series_db = self.get_time_series_database(database_index)
        batch_iterable = time_series_db.get_all(batch_size)
        for batch in batch_iterable:
            updated = []
            for time_series in batch:
                print("Calculating features for time series {0}".format(time_series.id)),
                time_series.calculate_features()
                updated.append(time_series)
                print("DONE")
            if len(updated) != 0:
                print("Updating values to database..."),
                time_series_db.update_many(updated)
                print("DONE")

    def reduce_all(self, serialization_db_index=0, reduction_model_index=0,
                   time_series_db_index=0, batch_size=None):
        time_series_db = self.get_time_series_database(time_series_db_index)
        batch_iterable = time_series_db.get_all(batch_size)
        reduction_model = self.get_reduction_model(serialization_db_index, reduction_model_index)
        serialization_db = self.get_serialization_database(serialization_db_index)
        for time_series_list in batch_iterable:
            print("Reducing dimensionality of {0} time series... ".format(len(time_series_list)))
            n_updated = reduction_model.add_transform_time_series(time_series_list)
            print("Updating {0} time series to database...".format(n_updated))
            if n_updated > 0:
                time_series_db.update_many(time_series_list)
                print("Updating reduction model to database...")
                serialization_db.store_reduction_model(reduction_model)
            print("DONE")

    def cluster_all(self, time_series_database_index=0,
                    serialization_db_index=0, clustering_model_index=0):
        time_series_db = self.get_time_series_database(time_series_database_index)
        clustering_model = self.get_clustering_model(serialization_db_index, clustering_model_index)
        serialization_db = self.get_serialization_database(serialization_db_index)
        batch_iterable = time_series_db.get_all()
        for time_series_list in batch_iterable:
            clustering_model.add_many_time_series(time_series_list)
        serialization_db.store_clustering_model(clustering_model)

    def store_all_clusters(self, time_series_db_index=0, clustering_db_index=0, serialization_db_index=0, clustering_model_index=0, ):
        clustering_model = self.get_clustering_model(serialization_db_index, clustering_model_index)
        clustering_db = self.get_clustering_database(clustering_db_index)
        time_series_db = self.get_time_series_database(time_series_db_index)
        clustering_db.reset_database()
        clusters_iterator = clustering_model.get_cluster_iterator(time_series_db)
        for i, cluster in itertools.izip(xrange(len(clusters_iterator)), clusters_iterator):
            clustering_db.store_cluster(i, clustering_db)



