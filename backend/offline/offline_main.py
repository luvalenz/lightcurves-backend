from backend.offline.offline_interface import OfflineInterface
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



if __name__ == "__main__":
    interface = OfflineInterface()
    #ts_db = interface.get_time_series_database()
    #transfer data
    #interface.transfer_time_series('macho', 1,0)
    #calculate features
    #interface.calculate_missing_features(0, 5)
    #interface.recalculate_all_features(0, 5)
    #setup
    #interface._setup()
    #reduce dimensionality
    #interface.reduce_all(0, 0, 0)
    #cluster
    # interface.cluster_all(0, 0, 0)
    # clustering_model = interface.get_clustering_model(0,0)
    # print(clustering_model)
    # mongodb = interface.get_time_series_database()
    # local_centers, local_clusters = clustering_model.get_cluster_list(mode='local')
    # plot_cluster_list(local_centers, local_clusters, mongodb)
    # global_centers, global_clusters = clustering_model.get_cluster_list(mode='global')
    # plot_cluster_list(global_centers, global_clusters, mongodb)
    interface.store_all_clusters()