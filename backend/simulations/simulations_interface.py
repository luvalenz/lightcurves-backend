__author__ = 'lucas'


from backend.data_model.time_series import DataMultibandTimeSeries
from backend.data_model.data_model_interface import DataModelInterface, load_config
from backend.online.online_algorithms import OurMethod
import matplotlib.pyplot as plt
import json
import random
import numpy as np
import cPickle as pickle
import os


class BirchSimulationsInterface(object):

    def __init__(self, data_model_interface):
        self._clustering_db = data_model_interface.get_clustering_database()
        self.time_series_db = data_model_interface.get_time_series_database()
        self._indexing_model = OurMethod(self._clustering_db, self.time_series_db, True)
        self._reduction_model = data_model_interface.get_reduction_model()

    @property
    def count(self):
        return self._clustering_db.data_points_count

    @property
    def clusters_count(self):
        return self._clustering_db.clusters_count

    @property
    def birch_radius(self):
        return self._clustering_db.metadata['radius']

    @staticmethod
    def get_macho_training_set_random_sample(field, quantity):
        with open('../macho_training_set_filenames.json', 'r') as training_ids_file:
            training_ids = json.load(training_ids_file)[str(field)]
            return random.sample(training_ids, quantity)

    def feature_space_query(self, feature_dict, n_neighbours):
        time_series_target = DataMultibandTimeSeries.from_dict({'features': feature_dict})
        self._reduction_model.transform_one_time_series(time_series_target)
        return self._indexing_model.time_series_query(time_series_target, n_neighbours)

    def time_series_space_query(self, time_series):
        pass

    def do_simulation(self, n_time_series, k_neighbors, name, training_set=False, field=None):
        n_bins = min([int(n_time_series), 50])
        if training_set:
            random_ids = BirchSimulationsInterface\
                .get_macho_training_set_random_sample(field, n_time_series)
            random_time_series = self\
                .time_series_db.get_many(random_ids, None, False, None, False, True)
        else:
            random_time_series = self.time_series_db.get_many_random(n_time_series)
        step1_calc_times = []
        step2_load_data_times = []
        step2_calc_times = []
        n_data_after_filters = []
        n_visited_clusters_list = []
        for time_series in random_time_series:
            retrieved_time_series, retrieved_distances, step1_calc_time,  \
                step2_load_data_time, step2_calc_time,\
                n_data_after_filter, n_visited_clusters = self.feature_space_query(time_series.feature_dict, k_neighbors)
            step1_calc_times.append(step1_calc_time)
            step2_load_data_times.append(step2_load_data_time)
            step2_calc_times.append(step2_calc_time)
            n_data_after_filters.append(n_data_after_filter)
            n_visited_clusters_list.append(n_visited_clusters)
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
        ax1.hist(step1_calc_times, n_bins)
        ax1.set_xlabel('Step 1 calculation time (seconds)')
        ax1.set_ylabel('Frequency')
        ax2.hist(step2_load_data_times, n_bins)
        ax2.set_xlabel('Step 2 database retrieval time (seconds)')
        ax2.set_ylabel('Frequency')
        ax3.hist(step2_calc_times, n_bins)
        ax3.set_xlabel('Step 2 calculation time (seconds)')
        ax3.set_ylabel('Frequency')
        ax4.hist(n_data_after_filters, n_bins)
        ax4.set_xlabel('Number of data points loaded')
        ax4.set_ylabel('Frequency')
        ax5.hist(n_visited_clusters_list, n_bins)
        ax5.set_xlabel('Number of clusters loaded')
        ax5.set_ylabel('Frequency')
        ax6.boxplot([step1_calc_times, step2_load_data_times,
                     step2_calc_times])
        ax6.set_xticklabels(['Step 1 calc.', 'Step 2 db retrieval', 'Step 2 calc.'])
        ax6.set_ylabel('Time (seconds)')
        # xtickNames = plt.setp(ax6, xticklabels=[5,6,7])
        # plt.setp(xtickNames, rotation=45, fontsize=8)

        print(step1_calc_times)
        print(step2_load_data_times)
        print(step2_calc_times)
        print(n_data_after_filters)
        print(n_visited_clusters_list)
        f.set_size_inches(18.5, 10.5)
        f.suptitle("Simulation with {0} targets\nn data points: {1},  "
                   "n clusters: {2},  birch radius: {3}".format(n_time_series, self.count,
                                                               self.clusters_count, self.birch_radius), fontsize=16)
        plt.savefig("{0}.jpg".format(name))
        plt.show()




    def plot_random(self, quantity):
        random_time_series = self.time_series_db.get_many_random(quantity)
        for time_series in random_time_series:
            plot_time_series(time_series)


class SimulationData(object):

    def __init__(self, name, step1_calc_times, step2_load_data_times,
                 step2_calc_times, n_data_after_filter,
                 n_visited_clusters_list, clusters_count, clustering_dbs_names, data_points_count):
        self.name = name
        self._step1_calc_times = step1_calc_times
        self._step2_load_data_times = step2_load_data_times
        self._step2_calc_times = step2_calc_times
        self._n_data_after_filter = n_data_after_filter
        self._n_visited_clusters_list = n_visited_clusters_list
        self._clusters_count = clusters_count
        self._clustering_dbs_names = clustering_dbs_names
        self._data_points_count = data_points_count

    @property
    def n_dbs(self):
        return len(self._step1_calc_times)

    @property
    def n_targets(self):
        return len(self._step1_calc_times[0])

    def plot_all(self, n_rows, n_bins=50, save=False):
        if save:
            self.save_pickle()
            if not os.path.exists(self.name):
                os.makedirs(self.name)
        self._plot_step1_calc(n_rows, n_bins, save)
        self.plot_step2_load_data(n_rows, n_bins, save)
        self.plot_step2_calc(n_rows, n_bins, save)
        self.plot_n_data_after_filter(n_rows, n_bins, save)
        self.plot_n_visited_clusters_list(n_rows, n_bins, save)
        self.plot_n_clusters_count(n_rows, n_bins, save)
        self.plot_times(save)
        self.plot_fetched_data(save)
        self.plot_data_distribution(save)

    def _plot_step1_calc(self, n_rows, n_bins, save=False):
        self._plot_hist_panel(n_rows, n_bins, save, self._step1_calc_times, "Step 1 calculations")

    def plot_step2_load_data(self, n_rows, n_bins, save=False):
        self._plot_hist_panel(n_rows, n_bins, save, self._step2_load_data_times, "Step 2 data fetching time")

    def plot_step2_calc(self, n_rows, n_bins, save=False):
        self._plot_hist_panel(n_rows, n_bins, save, self._step2_calc_times, "Step 2 calculations")

    def plot_n_data_after_filter(self, n_rows, n_bins, save=False):
        self._plot_hist_panel(n_rows, n_bins, save, self._n_data_after_filter, "Number of data points fetched")

    def plot_n_visited_clusters_list(self, n_rows, n_bins, save=False):
        self._plot_hist_panel(n_rows, n_bins, save, self._n_visited_clusters_list, "Number of clusters fetched")

    def plot_n_clusters_count(self, n_rows, n_bins, save=False):
        self._plot_hist_panel(n_rows, n_bins, save, self._clusters_count, "Clusters distribution")

    def _plot_hist_panel(self, n_rows, n_bins, save, values, plot_name):
        n_cols = int(np.ceil(float(self.n_dbs) / n_rows))
        f, axes = plt.subplots(n_rows, n_cols)
        axes_flat = [item for sublist in axes for item in sublist]
        for ax, v, clustering_db_name in zip(axes_flat, values, self._clustering_dbs_names):
            n_bins = min(n_bins, len(v))
            r = None
            if len(v) == 1:
                r = (v[0] - 1, v[0] + 1)
            ax.hist(v, bins=n_bins, range=r)
            ax.set_xlabel(clustering_db_name)
            ax.set_ylabel('Frequency')
            #ax.tick_params(axis='x', pad=10)
        for ax in axes_flat:
            plt.sca(ax)
            locs, labels = plt.xticks()
            plt.setp(labels, rotation=45)
        f.set_size_inches(18.5, 10.5)
        f.suptitle("Simulation {0}\ntargets: {1},   data points: {2}\n"
                   "{3}".format(self.name, self.n_targets,
                                self._data_points_count, plot_name), fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        if save:
            plt.savefig(os.path.join(self.name, '{0}.jpg'.format(plot_name)))
        else:
            plt.show()

    def plot_times(self, save=False):
        f, (ax0, ax1, ax2) = plt.subplots(1, 3)
        ax0.boxplot(self._step1_calc_times)
        ax0.set_xticklabels(self._clustering_dbs_names, rotation=45)
        ax0.set_ylabel('Time (seconds)')
        ax0.set_title('Step 1 calculations')
        ax1.boxplot(self._step2_load_data_times)
        ax1.set_xticklabels(self._clustering_dbs_names, rotation=45)
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Step 2 data fetching time')
        ax2.boxplot(self._step2_calc_times)
        ax2.set_xticklabels(self._clustering_dbs_names, rotation=45)
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Step 2 calculations')
        f.suptitle("Simulation {0}\ntargets: {1},   data points: {2}\n"
                   "Execution times".format(self.name, self.n_targets,
                                self._data_points_count), fontsize=16)
        f.set_size_inches(18.5, 10.5)
        plt.subplots_adjust(top=0.85)
        if save:
            plt.savefig(os.path.join(self.name, 'execution_times.jpg'))
        else:
            plt.show()

    def plot_fetched_data(self, save=False):
        f, (ax0, ax1) = plt.subplots(1, 2)
        ax0.boxplot(self._n_data_after_filter)
        ax0.set_xticklabels(self._clustering_dbs_names, rotation=45)
        ax0.set_ylabel('Count')
        ax0.set_title('Clusters fetched')
        ax1.boxplot(self._n_visited_clusters_list)
        ax1.set_xticklabels(self._clustering_dbs_names, rotation=45)
        ax1.set_ylabel('Count')
        ax1.set_title('Data points fetched')
        f.suptitle("Simulation {0}\ntargets: {1},   data points: {2}\n"
                   "Fetched data".format(self.name, self.n_targets,
                                self._data_points_count), fontsize=16)
        f.set_size_inches(18.5, 10.5)
        plt.subplots_adjust(top=0.85)
        if save:
            plt.savefig(os.path.join(self.name, 'fetched_data.jpg'))
        else:
            plt.show()

    def plot_data_distribution(self, save=False):
        n_data_per_cluster = [len(cluster) for cluster in self._clusters_count]
        f, (ax0, ax1) = plt.subplots(1, 2)
        ax0.boxplot(self._clusters_count)
        ax0.set_xticklabels(self._clustering_dbs_names, rotation=45)
        ax0.set_ylabel('Data per cluster')
        ax1.scatter(range(len(n_data_per_cluster)), n_data_per_cluster, marker="o")
        ax1.plot(range(len(n_data_per_cluster)), n_data_per_cluster, '.')
        ax1.set_xticks(range(len(n_data_per_cluster)))
        ax1.set_xticklabels(self._clustering_dbs_names, rotation=45)
        ax1.set_ylabel('Number of clusters')
        f.suptitle("Simulation {0}\ntargets: {1},   data points: {2}\n"
                   "Data distribution".format(self.name, self.n_targets,
                                self._data_points_count), fontsize=16)
        f.set_size_inches(18.5, 10.5)
        plt.subplots_adjust(top=0.85)
        if save:
            plt.savefig(os.path.join(self.name, 'data_distribution.jpg'))
        else:
            plt.show()

    def save_pickle(self):
        with open('{0}.pkl'.format(self.name), 'wb') as f:
            pickle.dump(self, f, protocol=2)


class BirchMultiDatabaseSimulationsInterface(object):

    def __init__(self, name, data_model_interface,
                 time_series_db_index, clustering_dbs_indices, reduction_model_index):
        self.name = name
        self._clustering_dbs = [data_model_interface.get_clustering_database(i) for i in clustering_dbs_indices]
        self._time_series_db = data_model_interface.get_time_series_database(time_series_db_index)
        self._indexing_models = [OurMethod(clustering_db, self._time_series_db, True)
                                 for clustering_db in self._clustering_dbs]
        self._reduction_model = data_model_interface.get_reduction_model(reduction_model_index)
        self._n_dbs = len(clustering_dbs_indices)
        self.simulatioin_data = None

    @property
    def count(self):
        return self._clustering_dbs[0].data_points_count

    def feature_space_query(self, feature_dict, n_neighbours, indexing_model):
        time_series_target = DataMultibandTimeSeries.from_dict({'features': feature_dict})
        self._reduction_model.transform_one_time_series(time_series_target)
        return indexing_model.time_series_query(time_series_target, n_neighbours)

    def _do_one_simulation(self, n_time_series, k_neighbors, indexing_model,
                          clustering_db, training_set=False, field=None):
        if training_set:
            random_ids = BirchSimulationsInterface\
                .get_macho_training_set_random_sample(field, n_time_series)
            random_time_series = self\
                ._time_series_db.get_many(random_ids, None, False, None, False, True)
        else:
            random_time_series = self._time_series_db.get_many_random(n_time_series)
        step1_calc_times = []
        step2_load_data_times = []
        step2_calc_times = []
        n_data_after_filters = []
        n_visited_clusters_list = []
        for time_series in random_time_series:
            retrieved_time_series, retrieved_distances, step1_calc_time,  \
                step2_load_data_time, step2_calc_time,\
                n_data_after_filter, n_visited_clusters = \
                self.feature_space_query(time_series.feature_dict, k_neighbors, indexing_model)
            step1_calc_times.append(step1_calc_time)
            step2_load_data_times.append(step2_load_data_time)
            step2_calc_times.append(step2_calc_time)
            n_data_after_filters.append(n_data_after_filter)
            n_visited_clusters_list.append(n_visited_clusters)
        clusters_count = clustering_db.counts.tolist()
        return step1_calc_times, step2_load_data_times,\
               step2_calc_times, n_data_after_filters, n_visited_clusters_list, clusters_count

    def do_all_simulations(self, n_time_series, k_neighbors):
        all_step1_calc_times = []
        all_step2_load_data_times = []
        all_step2_calc_times = []
        all_n_data_after_filters = []
        all_n_visited_clusters_list = []
        all_clusters_count = []
        for indexing_model, clustering_db in zip(self._indexing_models, self._clustering_dbs):
            print("Running simulations over {0} database".format(clustering_db.db_name))
            step1_calc_times, step2_load_data_times,\
            step2_calc_times, n_data_after_filters,\
            n_visited_clusters_list, clusters_count = self._do_one_simulation(
                n_time_series, k_neighbors, indexing_model, clustering_db)
            all_step1_calc_times.append(step1_calc_times)
            all_step2_load_data_times.append(step2_load_data_times)
            all_step2_calc_times.append(step2_calc_times)
            all_n_data_after_filters.append(n_data_after_filters)
            all_n_visited_clusters_list.append(n_visited_clusters_list)
            all_clusters_count.append(clusters_count)
        self.simulatioin_data = SimulationData(self.name, all_step1_calc_times, all_step2_load_data_times,
                                  all_step2_calc_times, all_n_data_after_filters,
                                  all_n_visited_clusters_list, all_clusters_count,
                                  [cluster_db.db_name for cluster_db in self._clustering_dbs], self.count)



def plot_time_series(time_series):
    time_series.fold()
    if 'B' in time_series:
        plt.plot(time_series['B'].phase, time_series['B'].values, '*', color='blue')
    if 'R' in time_series:
        plt.plot(time_series['R'].phase, time_series['R'].values, '*', color='red')
    plt.show()


def run_f1_simulation():
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    simulations_interface = BirchSimulationsInterface(data_model_interface)
    simulations_interface.do_simulation(1000, 10, 'field 1 radius 1')


def run_f1_simulation_training_set():
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    simulations_interface = BirchSimulationsInterface(data_model_interface)
    simulations_interface.do_simulation(1000, 10, 'field 1 radius 1 training set', True, 1)

def run_t1_t9_simulation():
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    clustering_dbs_indices = range(9)
    simulations_interface = BirchMultiDatabaseSimulationsInterface('Macho Field 1, radii 1.0 to 9.0',
                                                                   data_model_interface, 0, clustering_dbs_indices, 0)
    simulations_interface.do_all_simulations(100, 10)
    simulations_interface.simulatioin_data.plot_all(2, 50, True)

if __name__ == '__main__':
    print run_t1_t9_simulation()



