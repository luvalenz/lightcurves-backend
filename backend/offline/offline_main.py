import sys
sys.path.append('~/PycharmProjects/lightcurves-backend')

from backend.offline.offline_interface import OfflineInterface
from backend.data_model.data_model_interface import DataModelInterface, load_config
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma
from backend.offline.offline_algorithms import Birch
import time

def humanize_time(total_seconds):
    total_seconds = int(total_seconds)
    hours = total_seconds / 3600
    remainder = total_seconds % 3600
    minutes = remainder / 60
    seconds = remainder % 60
    return "{0}:{1}:{2}".format(hours, minutes, seconds)

def transfer_upto_field_1():
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    offline_interface = OfflineInterface(data_model_interface, 0, 0, 0, 0, 0)
    source_db_index = 1
    n_fields = 1
    offline_interface.fit_reduce_from_external_db('macho', source_db_index, n_fields)
    # source_db = data_model_interface.get_time_series_database(source_db_index)
    # offline_interface.reduce_from_external_db('macho', 0, source_db, n_fields)

def cluster(clustering_model_index, last_field, time_series_db_index):
    print ("### CLUSTERING FIELD 1 WITH model {0} ###".format(clustering_model_index))
    config = load_config('/n/home09/lvalenzuela/lightcurves-backend/backend/config.json')
    #config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    serialization_db_index = clustering_model_index
    offline_interface = OfflineInterface(data_model_interface, time_series_db_index, 0, serialization_db_index, clustering_model_index, 0)
    offline_interface.cluster_all()


def store_field1_unbalanced():
    clustering_db_indices = range(43,50)
    for clustering_db_index in clustering_db_indices:
        print ("### STORING FIELD 1 TO DB {0} ###".format(clustering_db_index))
        config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
        data_model_interface = DataModelInterface(config)
        time_series_db_index = 3
        serialization_db_index = clustering_db_index
        clustering_model_index = clustering_db_index
        reduction_model_index = 0
        offline_interface = OfflineInterface(data_model_interface, time_series_db_index, clustering_db_index,
             serialization_db_index, clustering_model_index, reduction_model_index)
        print(offline_interface.serialization_db._name)
        offline_interface.store_all_clusters()

def store_field1_balanced():
    dbs = range(56,100)
    max_radii = []
    for i in range(4):
        max_radii += np.linspace(0, 250*20**i, num=11)[1:].astype(int).tolist()
    for clustering_db_index, max_size in zip(dbs, max_radii):
        print ("### STORING FIELD 1 TO DB {0} ###".format(clustering_db_index))
        config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
        data_model_interface = DataModelInterface(config)
        time_series_db_index = 3
        serialization_db_index = 0
        clustering_model_index = 0
        reduction_model_index = 0
        offline_interface = OfflineInterface(data_model_interface, time_series_db_index, clustering_db_index,
             serialization_db_index, clustering_model_index, reduction_model_index)
        offline_interface.store_all_clusters(max_size)


def clustering_test():
    n = 1000
    d = 2
    x1 = 100*np.random.uniform(-1,1,(n*d)).reshape((n,d))
    x2 = 1*np.random.uniform(-1,1,(n*d)).reshape((n,d))
    x = np.vstack((x1,x2))
    brc = Birch(0.5, False, 2)
    for id_, element in zip(range(len(x)), x):
        brc._add_data_point(str(id_), element)
  #  brc.clusters_max_size = 2
    n_clusters = brc.get_number_of_clusters()
    l = brc.get_cluster_list()
    print(n_clusters)
    print(l)

def to_pands():
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    offline_interface = OfflineInterface(data_model_interface, 0, 0, 0, 0, 0)
    offline_interface.to_pandas_dataframe('/home/lucas/pandas_db','lightcurves_f1')

if __name__ == "__main__":
    start = time.time()
    clustering_model_index = int(sys.argv[1])
    last_field = int(sys.argv[2])
    time_series_db_index = int(sys.argv[3])
    cluster(clustering_model_index, last_field, time_series_db_index)
    #store_field1_balanced()
    # to_pands()
    end = time.time()
    print(humanize_time(end-start))
