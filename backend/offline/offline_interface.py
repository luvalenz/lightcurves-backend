__author__ = 'lucas'

from backend.data_model.data_model_interface import DataModelInterface
import itertools


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
        return  self._clustering_model

    @property
    def reduction_model(self):
        if self._reduction_model is None:
            self._reduction_model = self._data_model_interface.\
                get_reduction_model(self._serialization_db_index, self._reduction_model_index)

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
        batch_iterable = self.time_series_db.get_all(batch_size)
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
    
    def reduce_all(self, batch_size=None):
        batch_iterable = self.time_series_db.get_all(batch_size)
        for time_series_list in batch_iterable:
            print("Trying to add {0} time series to reduction model... ".format(len(time_series_list))),
            n_updated = self.reduction_model.add_many_time_series(time_series_list)
            print("DONE. {0} added successfully.".format(n_updated))
        print("Updating reduction model to database..."),
        self.serialization_db.store_reduction_model(self.reduction_model)
        print("DONE")
        batch_iterable.rewind()
        for time_series_list in batch_iterable:
            print("Trying to reduce dimensionality of {0} time series... ".format(len(time_series_list))),
            n_updated = self.reduction_model.transform_time_series(time_series_list)
            print("DONE")
            if n_updated > 0:
                print("Updating {0} time series to database...".format(n_updated)),
                self.time_series_db.update_many(time_series_list)
                print("DONE")

    def reduce_some(self, time_series_ids, catalog_name=None, batch_size=None):
        batch_iterable = self.time_series_db.get_many(time_series_ids, catalog_name, True, batch_size)
        for time_series_list in batch_iterable:
            print("Trying to reduce dimensionality of {0} time series... ".format(len(time_series_list))),
            n_updated = self.reduction_model.add_transform_time_series(time_series_list)
            print("DONE")
            if n_updated > 0:
                print("Updating {0} time series to database...".format(n_updated))
                self.time_series_db.update_many(time_series_list)
                print("DONE")
                print("Updating reduction model to database..."),
                self.serialization_db.store_reduction_model(self.reduction_model)
            print("DONE")
    
    def cluster_all(self):
        batch_iterable = self.time_series_db.get_all()
        for time_series_list in batch_iterable:
            print("Trying to add {0} time series to clustering model... ".format(len(time_series_list))),
            n_added = self.clustering_model.add_many_time_series(time_series_list)
            print("{0} added.".format(n_added))
        print("Updating clustering model to database..."),
        self.serialization_db.store_clustering_model(self.clustering_model)
        print("DONE")
    
    def cluster_some(self, time_series_ids, catalog_name=None):
        batch_iterable = self.time_series_db.get_many(time_series_ids, catalog_name)
        for time_series_list in batch_iterable:
            print("Trying to add {0} time series to clustering model... ".format(len(time_series_list))),
            n_added = self.clustering_model.add_many_time_series(time_series_list)
            print("{0} added.".format(n_added))
        print("Updating clustering model to database..."),
        self.serialization_db.store_clustering_model(self.clustering_model)
        print("DONE")
    
    def store_all_clusters(self):
        self.clustering_db.reset_database()
        clusters_iterator = self.clustering_model.get_cluster_iterator(self.time_series_db)
        for i, cluster in itertools.izip(xrange(len(clusters_iterator)), clusters_iterator):
            self.clustering_db.store_cluster(i, cluster)
    
    def get_clusters(self):
        return self.clustering_db.get_all_clusters(False)




