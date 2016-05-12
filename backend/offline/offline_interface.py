__author__ = 'lucas'

from backend.data_model.data_model_interface import DataModelInterface
import itertools
import numpy as np
import pandas as pd


class OfflineInterface(object):
    
    def __init__(self, data_model_interface, time_series_db_index=0, clustering_db_index=0,
                 serialization_db_index=0, clustering_model_index=0, reduction_model_index=0):
        self._data_model_interface = data_model_interface
        self._time_series_db_index = time_series_db_index
        self._clustering_db_index = clustering_db_index
        self._serialization_db_index = serialization_db_index
        self._clustering_model_index = clustering_model_index
        self._reduction_model_index = reduction_model_index
        self._time_series_db = None
        self._clustering_db = None
        self._serialization_db = None
        self._clustering_model = None
        self._reduction_model = None
    
    @property
    def time_series_db(self):
        if self._time_series_db is None:
            self._time_series_db = self._data_model_interface.get_time_series_database(self._time_series_db_index)
        return self._time_series_db
    
    @property
    def clustering_db(self):
        if self._clustering_db is None:
            self._clustering_db = self._data_model_interface.get_clustering_database(self._clustering_db_index)
        return self._clustering_db
    
    @property
    def serialization_db(self):
        if self._serialization_db is None:
            self._serialization_db = self._data_model_interface.\
                get_serialization_database(self._serialization_db_index)
        return self._serialization_db

    @property
    def clustering_model(self):
        if self._clustering_model is None:
            self._clustering_model = self._data_model_interface.get_clustering_model\
                (self._serialization_db_index, self._clustering_model_index)
        return self._clustering_model

    @property
    def reduction_model(self):
        if self._reduction_model is None:
            self._reduction_model = self._data_model_interface.\
                get_reduction_model(self._serialization_db_index, self._reduction_model_index)
        return self._reduction_model

    #bust be called after adding elements and before reduction
    def setup(self):
        self.time_series_db.setup()
        self.clustering_db.setup()
        self.serialization_db.setup()
    
    def transfer_time_series(self, catalog_name, source_database_index):
        source_db = self._data_model_interface.get_time_series_database(source_database_index)
        destination_db = self.time_series_db
        batch_iterable = source_db.get_all()
        added_ids = []
        for batch in batch_iterable:
            if len(batch) != 0:
                print ('Adding {0} elements to destination...'.format(len(batch))),
                added_ids += destination_db.add_many(catalog_name, batch)
                print('DONE')
        self.setup()
        return added_ids

    def fit_reduction_with_external_db(self, source_db, n_fields):
        batch_iterable = source_db.get_all(n_fields, False)
        total_added = 0
        for batch in batch_iterable:
            if len(batch) != 0:
                print("Trying to add {0} time series to reduction model... ".format(len(batch))),
                n_added = self.reduction_model.add_many_time_series(batch)
                total_added += n_added
                print("DONE, {0} added".format(n_added))
        print("Fitting model... "),
        self.reduction_model.fit()
        print("DONE")
        print("\nUpdating reduction model to database..."),
        self.serialization_db.store_reduction_model(self.reduction_model)
        print("DONE\n")
        return total_added

    def reduce_from_external_db(self, catalog_name, total_added, source_db, n_fields):
        batch_iterable = source_db.get_missing(n_fields, self.time_series_db, False)
        for batch in batch_iterable:
            if batch is not None:
                n_data = len(batch)
                if n_data != 0:
                    print("Calculating reduced vectors for {0} time series... ".format(n_data)),
                    updated_time_series_iterator = self.reduction_model.transform_many_time_series(batch)
                    print('DONE')
                    if updated_time_series_iterator is not None:
                        print("Adding time series to database..."),
                        self.time_series_db.add_many(catalog_name, updated_time_series_iterator)
                        print("DONE")

    def fit_reduce_from_external_db(self, catalog_name, source_database_index, n_fields):
        source_db = self._data_model_interface.get_time_series_database(source_database_index)
        n_added = self.fit_reduction_with_external_db(source_db, n_fields)
        if n_added > 0:
            self.reduce_from_external_db(catalog_name, n_added, source_db, n_fields)

    def transfer_time_series_conditionally(self, catalog_name, source_database_index, last_field):
        source_db = self._data_model_interface.get_time_series_database(source_database_index)
        destination_db = self.time_series_db
        batch_iterable = source_db.get_missing(last_field, destination_db)
        added_ids = []
        for batch in batch_iterable:
            if batch is not None and len(batch) != 0:
                print("Trying to add {0} time series to reduction model... ".format(len(batch))),
                n_added = self.reduction_model.add_many_time_series(batch)
                print("DONE, {0} added".format(n_added))
        self.setup()
        return added_ids
    
    def defragment_clusters(self):
        self.clustering_db.defragment()
    
    def calculate_missing_features(self, batch_size=None):
        batch_iterable = self.time_series_db.find_many(None, {"features":{"$in": [{}, None]}}, True, batch_size)
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
                self.time_series_db.update_many(updated)
                print("DONE")
    
    def recalculate_all_features(self, batch_size=None):
        batch_iterable = self.time_series_db.get_all(True, batch_size)
        for batch in batch_iterable:
            updated = []
            for time_series in batch:
                print("Calculating features for time series {0}".format(time_series.id)),
                time_series.calculate_features()
                updated.append(time_series)
                print("DONE")
            if len(updated) != 0:
                print("Updating values to database..."),
                self.time_series_db.update_many(updated)
                print("DONE")

    def _reduce(self, time_series_iterable):
        print("Trying to add {0} time series to reduction model... ".format(len(time_series_iterable))),
        n_added = self.reduction_model.add_many_time_series(time_series_iterable)
        print("DONE, {0} added".format(n_added))
        print("Fitting model... "),
        self._reduction_model.fit()
        print("DONE")
        print("Updating reduction model to database..."),
        self.serialization_db.store_reduction_model(self.reduction_model)
        print("DONE")
        if n_added > 0:
            print("Calculating reduced vectors for {0} time series... ".format(n_added)),
            all_time_series_iterator = self.time_series_db.get_all()
            updated_time_series_iterator = self._reduction_model.transform_many_time_series(all_time_series_iterator)
            print("DONE")
            print("Updating all time series to database...")
            self.time_series_db.update_many(updated_time_series_iterator, True)
            print("DONE")

    def reduce_all(self):
        time_series_iterator = self.time_series_db.get_all()
        self._reduce(time_series_iterator)

    def reduce_some(self, time_series_ids, catalog_name=None):
        time_series_iterator = self.time_series_db.get_many(time_series_ids, catalog_name, False)
        self._reduce(time_series_iterator)

    def _cluster(self, time_series_sequence):
        print("Trying to add time series to clustering model... "),
        n_added = self.clustering_model.add_many_time_series(time_series_sequence)
        print("{0} successfully added".format(n_added))
        print("Updating clustering model to database..."),
        self.serialization_db.store_clustering_model(self.clustering_model)
        print("DONE")
    
    def cluster_all(self):
        time_series_sequence = self.time_series_db.get_all()
        self._cluster(time_series_sequence)

    def to_pandas_dataframe(self, path):
        reduced_vectors = []
        ids = []
        time_series_sequence = self.time_series_db.get_all()
        i = 0
        for time_series in time_series_sequence:
            reduced_vector = time_series.reduced_vector
            if reduced_vector is not None and len(reduced_vector) != 0:
                ids.append(time_series.id)
                reduced_vectors.append(reduced_vector)
            if i % 1000 == 0:
                print(i)
            i += 1
        reduced_matrix = np.vstack(reduced_vectors)
        df = pd.DataFrame(reduced_matrix, ids)
        df.to_pickle(path)


    def cluster_some(self, time_series_ids, catalog_name=None):
        batch_iterable = self.time_series_db.get_many(time_series_ids, catalog_name, False)
        self._cluster(batch_iterable)
    
    def store_all_clusters(self, clusters_max_size=None):
        if clusters_max_size is not None:
            self.clustering_model.clusters_max_size = clusters_max_size
        n_clusters = int(self.clustering_model.get_number_of_clusters())
        self.clustering_db.reset_database(self.clustering_model.metadata)
        print("Storing {0} clusters...".format(n_clusters))
        clusters_iterator = self.clustering_model.get_cluster_iterator(self.time_series_db)
        print("Iterator length: {0}".format(len(clusters_iterator)))
        for i, cluster in itertools.izip(xrange(len(clusters_iterator)), clusters_iterator):
            self.clustering_db.store_cluster(i, cluster)
            if i > 0 and i % 500 == 0:
                print "{0} clusters stored".format(i)
    
    def get_clusters(self):
        return self.clustering_db.get_all_clusters(False)




