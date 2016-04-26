from backend.offline.offline_interface import OfflineInterface
from backend.data_model.data_model_interface import DataModelInterface, load_config
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma

import time


def extract_feature_matrix(database, id_list):
    time_series_iterator = database.get_many(id_list, 'macho', False)
    feature_vectors = []
    for time_series in time_series_iterator:
        feature_vector = time_series.reduced_vector
        if len(feature_vector) != 0:
            feature_vectors.append(feature_vector)
    #    else:
    #        print(time_series.id)
    #print('{0}, {1}'.format(len(id_list), len(feature_vectors)))
    return np.array((feature_vectors))


def plot_cluster_list(centers, clusters, database):
    plt.plot(centers[:, 0], centers[:, 1], 'x')
    colors = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))
    np.random.shuffle(colors)
    for cluster_indices, col in zip(clusters, colors):
        cluster_data = extract_feature_matrix(database, cluster_indices)
        plt.plot(cluster_data[:, 0], cluster_data[:, 1], 'o', markerfacecolor=col)
    plt.show()


def plot_clusters(clusters):
    colors = plt.cm.Spectral(np.linspace(0, 1, len(clusters)))
    np.random.shuffle(colors)
    for cluster, col in zip(clusters, colors):
        cluster_data = cluster.data_points
        plt.plot(cluster_data[:, 0], cluster_data[:, 1], 'o', markerfacecolor=col)
    plt.show()


def humanize_time(total_seconds):
    total_seconds = int(total_seconds)
    hours = total_seconds / 3600
    remainder = total_seconds % 3600
    minutes = remainder / 60
    seconds = remainder % 60
    return "{0}:{1}:{2}".format(hours, minutes, seconds)


def cluster_field1():
    for clustering_model in range(13, 19):
        threshold = float(clustering_model - 9)/10
        print ("### CLUSTERING FIELD 1 WITH THRESHOLD {0} ###".format(threshold))
        config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
        data_model_interface = DataModelInterface(config)
        offline_interface = OfflineInterface(data_model_interface, 0, clustering_model, clustering_model, clustering_model, 0)
        offline_interface.cluster_all()


def cluster_field1_2():
    print ("### CLUSTERING FIELD 1 WITH THRESHOLD 1.15 ###")
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    offline_interface = OfflineInterface(data_model_interface, 0, 1, 19, 19, 0)
    #offline_interface.cluster_all()
    offline_interface.store_all_clusters()


def store_f1_t1():
    print ("### STORING FIELD 1 WITH THRESHOLD 1.0")
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    offline_interface = OfflineInterface(data_model_interface, 0, 0, 0, 0, 0)
    print offline_interface.clustering_model.get_number_of_clusters()
    print offline_interface.clustering_db._database.name
    offline_interface.store_all_clusters()


def store_f1_t_0_1_to_0_9():
    for i in range(10, 19):
        threshold = float(i - 9)/10
        print ("### STORING FIELD 1 WITH THRESHOLD {0}".format(threshold))
        config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
        data_model_interface = DataModelInterface(config)
        offline_interface = OfflineInterface(data_model_interface, 0, i, i, i, 0)
        print offline_interface.clustering_model.get_number_of_clusters()
        print offline_interface.clustering_db._database.name
        offline_interface.store_all_clusters()


def store_f1_t1_c10e4():
    print ("### STORING FIELD 1 WITH THRESHOLD 1.0 and 10000 aglomerative clusters")
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    offline_interface = OfflineInterface(data_model_interface, 0, 1, 19, 19, 0)
    print offline_interface.clustering_db._database.name
    print offline_interface.serialization_db.db.name
    print offline_interface.clustering_model.threshold
    print offline_interface.clustering_model._n_global_clusters
    print len(offline_interface.clustering_model.get_number_of_clusters())
    offline_interface.store_all_clusters()


def calculate_stats():
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    time_series_db = data_model_interface.get_time_series_database()
    time_series_sequence = time_series_db.get_all()
    i = 0
    reduced_vectors = []
    for time_series in time_series_sequence:
        reduced = time_series.reduced_vector
        if reduced is not None:
            reduced_vectors.append(reduced)
        if i % 1000 == 0:
            print "{0} scanned...".format(i)
        i += 1
    reduced_matrix = np.vstack(reduced_vectors)
    min_vector = np.min(reduced_matrix, axis=0)
    max_vector = np.max(reduced_matrix, axis=0)
    mean_vector = np.mean(reduced_matrix, axis=0)
    std_vector = np.std(reduced_matrix, axis=0)
    print("Min: {0}".format(min_vector))
    print("Max: {0}".format(max_vector))
    print("Mean: {0}".format(mean_vector))
    print("Std: {0}".format(std_vector))


def plot_n_clusters_vs_thresholds():
    d = 5
    min_f1 = np.array([-10.92337428, -119.23690121, -11.79905027, -92.11804735, -18.12778179])
    max_f1 = np.array([ 471.00212086, 20.8885065, 690.91728463, 215.06732147, 8.3393324])
    mean_f1 = np.array([5.67188158e-13, -3.15321667e-12, -1.58469618e-12, 2.08638346e-12, 7.16403364e-14])
    std_f1 = np.array([3.79032448, 2.57827277, 2.21436251, 1.5504707, 1.44090656])
    edges = max_f1 - min_f1
    edges_modified = 8*std_f1
    print(edges)
    print(edges_modified)

    c3 = lambda r, factor: np.prod(factor*std_f1)/(np.array(2*r))**d
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    thresholds = []
    n_clusters = []
    for i in range(20):
        try:
            offline_interface = OfflineInterface(data_model_interface, 0, 0, i, i, 0)
            birch = offline_interface.clustering_model
            n_clusters.append(birch.number_of_local_labels)
            thresholds.append(birch.threshold)
            print "{0}, {1}".format(birch.threshold, birch.number_of_local_labels)
        except ValueError:
            pass
    domain = np.arange(0, 10, 0.05)
    print thresholds
    print len(thresholds)
    print n_clusters
    print len(n_clusters)
    sigma_factor = 6
    plt.title("{0} sigma".format(sigma_factor))
    plt.plot(thresholds, n_clusters, 'o', color='blue')
    plt.plot(domain, c3(domain, sigma_factor), color='red')
    plt.show()
    # plt.plot(domain, c4(domain), color='black')




def transfer_field_1_to_2():
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    offline_interface = OfflineInterface(data_model_interface, 2, 0, 0, 0, 0)
    offline_interface.transfer_time_series('macho', 1)


if __name__ == "__main__":
    start = time.time()
    transfer_field_1_to_2()
    end = time.time()
    print humanize_time(end-start)
