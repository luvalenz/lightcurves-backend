from backend.offline.offline_interface import OfflineInterface
from backend.data_model.data_model_interface import DataModelInterface, load_config
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

import time


def humanize_time(total_seconds):
    total_seconds = int(total_seconds)
    hours = total_seconds / 3600
    remainder = total_seconds % 3600
    minutes = remainder / 60
    seconds = remainder % 60
    return "{0}:{1}:{2}".format(hours, minutes, seconds)


def transfer_field_1():
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    offline_interface = OfflineInterface(data_model_interface, 2, 0, 0, 0, 0)
    #offline_interface.fit_reduce_from_external_db('macho', 1, 1)
    source_db_index = 1
    source_db = data_model_interface.get_time_series_database(source_db_index)
    n_fields = 1
    offline_interface.reduce_from_external_db('macho', 0, source_db, n_fields)

def cluster_field1():
    for clustering_model in range(11, 20):
        if clustering_model % 10 != 0:
            print ("### CLUSTERING FIELD 1 WITH model {0} ###".format(clustering_model))
            config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
            data_model_interface = DataModelInterface(config)
            offline_interface = OfflineInterface(data_model_interface, 2, clustering_model, clustering_model, clustering_model, 0)
            offline_interface.cluster_all()

def store_field1_balanced():
    dbs = range(22, 40)
    max_radii = np.hstack((range(1,10), np.arange(0.1,1,0.1)))
    for clustering_db_index, max_size in zip(dbs, max_radii):
        if clustering_db_index % 10 != 0:
            print ("### STORING FIELD 1 TO DB {0} ###".format(clustering_db_index))
            config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
            data_model_interface = DataModelInterface(config)
            time_series_db_index = 2
            serialization_db_index = 11
            clustering_model_index = 11
            reduction_model_index = 0
            offline_interface = OfflineInterface(data_model_interface, time_series_db_index, clustering_db_index,
                 serialization_db_index, clustering_model_index, reduction_model_index)
            offline_interface.store_all_clusters(max_size)

# def cluster_field1_new_birch():
#     for clustering_model in range(13, 19):
#         threshold = float(clustering_model - 9)/10
#         print ("### CLUSTERING FIELD 1 WITH THRESHOLD {0} ###".format(threshold))
#         config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
#         data_model_interface = DataModelInterface(config)
#         offline_interface = OfflineInterface(data_model_interface, 0, clustering_model, clustering_model, clustering_model, 0)
#         offline_interface.cluster_all()


if __name__ == "__main__":
    start = time.time()
    store_field1_balanced()
    end = time.time()
    print(humanize_time(end-start))
