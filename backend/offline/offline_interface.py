__author__ = 'lucas'

import json
import glob
from backend.data_model.data_model_interface import DataModelInterface
import itertools


import os


class OfflineInterface(object):

    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __init__(self):
        pass

    #bust be called after adding elements and before reduction
    def _setup(self, time_series_db_index=0, clustering_db_index=0, serializing_db_index=0):
        time_series_db = DataModelInterface().get_time_series_database(time_series_db_index)
        clustering_db = DataModelInterface().get_clustering_database(clustering_db_index)
        serializing_db = DataModelInterface().get_serialization_database(serializing_db_index)
        time_series_db.setup()
        clustering_db.setup()
        serializing_db.setup()

    def transfer_time_series(self, catalog_name, source_database_index, destination_database_index=0):
        source_db = DataModelInterface().get_time_series_database(source_database_index)
        destination_db = DataModelInterface().get_time_series_database(destination_database_index)
        batch_iterable = source_db.get_all()
        added_ids = []
        for batch in batch_iterable:
            if len(batch) != 0:
                print ('adding {0} elements to destination...'.format(len(batch))),
                added_ids += destination_db.add_many(catalog_name, batch)
                print('DONE')
        self._setup()
        return added_ids

    def defragment_clusters(self, clustering_db_index=0):
        clustering_db = DataModelInterface().get_clustering_database(clustering_db_index)
        clustering_db.defragment()


    def calculate_missing_features(self, database_index=0, batch_size=None):
        time_series_db = DataModelInterface().get_time_series_database(database_index)
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
        time_series_db = DataModelInterface().get_time_series_database(database_index)
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
        time_series_db = DataModelInterface().get_time_series_database(time_series_db_index)
        batch_iterable = time_series_db.get_all(batch_size)
        reduction_model = DataModelInterface().get_reduction_model(serialization_db_index, reduction_model_index)
        serialization_db = DataModelInterface().get_serialization_database(serialization_db_index)
        for time_series_list in batch_iterable:
            print("Trying to reduce dimensionality of {0} time series... ".format(len(time_series_list))),
            n_updated = reduction_model.add_transform_time_series(time_series_list)
            print("DONE")
            if n_updated > 0:
                print("Updating {0} time series to database...".format(n_updated)),
                time_series_db.update_many(time_series_list)
                print("DONE")
                print("Updating reduction model to database..."),
                serialization_db.store_reduction_model(reduction_model)
                print("DONE")

    def reduce_some(self, time_series_ids, catalog_name=None, serialization_db_index=0,
                    reduction_model_index=0,time_series_db_index=0, batch_size=None):
        time_series_db = DataModelInterface().get_time_series_database(time_series_db_index)
        batch_iterable = time_series_db.get_many(time_series_ids, catalog_name, True, batch_size)
        reduction_model = DataModelInterface().get_reduction_model(serialization_db_index, reduction_model_index)
        serialization_db = DataModelInterface().get_serialization_database(serialization_db_index)
        for time_series_list in batch_iterable:
            print("Trying to reduce dimensionality of {0} time series... ".format(len(time_series_list))),
            n_updated = reduction_model.add_transform_time_series(time_series_list)
            print("DONE")
            if n_updated > 0:
                print("Updating {0} time series to database...".format(n_updated))
                time_series_db.update_many(time_series_list)
                print("DONE")
                print("Updating reduction model to database..."),
                serialization_db.store_reduction_model(reduction_model)
            print("DONE")

    def cluster_all(self, time_series_database_index=0,
                    serialization_db_index=0, clustering_model_index=0):
        time_series_db = DataModelInterface().get_time_series_database(time_series_database_index)
        clustering_model = DataModelInterface().get_clustering_model(serialization_db_index, clustering_model_index)
        serialization_db = DataModelInterface().get_serialization_database(serialization_db_index)
        batch_iterable = time_series_db.get_all()
        for time_series_list in batch_iterable:
            print("Trying to add {0} time series to clustering model... ".format(len(time_series_list))),
            n_added = clustering_model.add_many_time_series(time_series_list)
            print("{0} added.".format(n_added))
        print("Updating clustering model to database..."),
        serialization_db.store_clustering_model(clustering_model)
        print("DONE")

    def cluster_some(self, time_series_ids, catalog_name=None, time_series_database_index=0,
                    serialization_db_index=0, clustering_model_index=0):
        time_series_db = DataModelInterface().get_time_series_database(time_series_database_index)
        clustering_model = DataModelInterface().get_clustering_model(serialization_db_index, clustering_model_index)
        serialization_db = DataModelInterface().get_serialization_database(serialization_db_index)
        batch_iterable = time_series_db.get_many(time_series_ids, catalog_name)
        for time_series_list in batch_iterable:
            print("Trying to add {0} time series to clustering model... ".format(len(time_series_list))),
            n_added = clustering_model.add_many_time_series(time_series_list)
            print("{0} added.".format(n_added))
        print("Updating clustering model to database..."),
        serialization_db.store_clustering_model(clustering_model)
        print("DONE")

    def store_all_clusters(self, time_series_db_index=0, clustering_db_index=0,
                           serialization_db_index=0, clustering_model_index=0):
        clustering_model = DataModelInterface().get_clustering_model(serialization_db_index, clustering_model_index)
        clustering_db = DataModelInterface().get_clustering_database(clustering_db_index)
        time_series_db = DataModelInterface().get_time_series_database(time_series_db_index)
        clustering_db.reset_database()
        clusters_iterator = clustering_model.get_cluster_iterator(time_series_db)
        for i, cluster in itertools.izip(xrange(len(clusters_iterator)), clusters_iterator):
            clustering_db.store_cluster(i, cluster)

    def get_clusters(self, clustering_db_index):
        clustering_db = self.get_clustering_database(clustering_db_index)
        return clustering_db.get_all_clusters(False)




