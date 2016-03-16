from backend.offline.offline_interface import OfflineInterface
from backend.data_model.data_model_interface import DataModelInterface, load_config
import matplotlib.pyplot as plt
import numpy as np

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



if __name__ == "__main__":


    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    offline_interface = OfflineInterface(data_model_interface)

    # ids = offline_interface.transfer_time_series('macho', 1, 0)

    # #calculate features
    # #interface.calculate_missing_features(0, 5)
    # #interface.recalculate_all_features(0, 5)
    # #setup
    # offline_interface.setup()
    # #reduce dimensionality
    # offline_interface.reduce_all()
    # #cluster
    # offline_interface.cluster_all()
    # clustering_model = data_model_interface.get_clustering_model(0,0)
    # mongodb = data_model_interface.get_time_series_database()
    # local_centers, local_clusters = clustering_model.get_cluster_list(mode='local')
    # plot_cluster_list(local_centers, local_clusters, mongodb)
    # global_centers, global_clusters = clustering_model.get_cluster_list(mode='global')
    # #plot_cluster_list(global_centers, global_clusters, mongodb)
    offline_interface.store_all_clusters()
    #interface.get_clustering_model(0,0)

    # clustering_db = interface.get_clustering_database()
    # clusters_iterator = clustering_db.get_all()
    # plot_clusters(clusters_iterator)