__author__ = 'lucas'


from backend.data_model.time_series import DataMultibandTimeSeries
from backend.data_model.data_model_interface import DataModelInterface, load_config
from backend.online.online_algorithms import OurMethod
from scipy import stats
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


class RegressionData(object):

    def __init__(self, simulation_data):
        self._simulation_data = simulation_data
        self._log_seek_time_polyfit = None
        self._log_clusters_after_filter_polyfit = None
        self._log_clusters_after_filter_polyfit_2 = None
        self._transfer_time_polyfit = None
        self._step2_time_polyfit = None
        self._avg_data_points_per_cluster_slope = None
        self._avg_data_points_per_cluster_intercept = None
        self._avg_data_points_per_cluster_polyfit_2 = None
        self._clusters_polyfit = None
        self._log_fetched_data_polyfit = None
        self._avg_fetched_data_small_radii_slope = None

        self.step1_vs_n_clusters_polyfit = None
        self.step2_vs_fetched_data_polyfit = None


        self._do_all_regressions()

    @property
    def name(self):
        return self._simulation_data.name + "_regressions"

    @property
    def _avg_data_points_per_cluster_coefficient(self):
        return np.exp(self._avg_data_points_per_cluster_intercept)

    @property
    def _avg_data_points_per_cluster_exponent(self):
        return self._avg_data_points_per_cluster_slope
    #
    # @property
    # def _avg_fetched_data_small_radii_coefficient(self):
    #     return np.exp(self._avg_fetched_data_small_radii_intercept)
    #
    # # @property
    # def _avg_fetched_data_small_radii_exponent(self):
    #     return self._avg_fetched_data_small_radii_slope
    #
    # @property
    # def _avg_fetched_data_large_radii_coefficient(self):
    #     return np.exp(self._avg_fetched_data_large_radii_intercept)
    #
    # @property
    # def _avg_fetched_data_large_radii_exponent(self):
    #     return  self._avg_fetched_data_large_radii_slope

    def _do_all_regressions(self):
        self._do_clusters_regression()
        # self._do_seek_time_regression()
        # self._do_transfer_time_regression()
        self._do_clusters_after_filter_regression()
        self._do_avg_data_points_per_cluster_regresion()
        self._do_avg_data_points_per_cluster_regression_2()
        self._do_log_fetched_data_regression()
        self._do_step2_time_regression()
        self._do_clusters_after_filter_regression_2()
        self._do_step1_vs_n_clusters_regression()
        self._do_step2_vs_fetched_data_regression()

    def _do_step1_vs_n_clusters_regression(self):
        x = self._simulation_data.n_clusters
        y = self._simulation_data.mean_step1_times
        self.step1_vs_n_clusters_polyfit = np.polyfit(x, y, 1)

    def _do_step2_vs_fetched_data_regression(self):
        x = self._simulation_data.avg_fetched_data_points
        y = self._simulation_data.mean_step2_times
        self.step2_vs_fetched_data_polyfit = np.polyfit(x, y, 1)

    def _do_seek_time_regression(self):
        birch_radii = self._simulation_data.birch_radii
        mean_seek_times = self._simulation_data.mean_seek_times
        x = np.log(birch_radii)
        y = np.log(mean_seek_times)
        self._log_seek_time_polyfit = np.polyfit(x, y, 2)

    def _do_clusters_regression(self):
        x = self._simulation_data.birch_radii
        y = self._simulation_data.n_clusters
        x_rec = np.exp(-1*np.log(x))
        mask = np.where(np.logical_and(x <= 9, x >= 0.2))[0]
        x_rec_masked = x_rec[mask]
        y_masked = y[mask]
        self._clusters_polyfit = np.polyfit(x_rec_masked, y_masked, 2)

    def _do_transfer_time_regression(self):
        x = self._simulation_data.birch_radii
        y = self._simulation_data.mean_transfer_times
        self._transfer_time_polyfit = np.polyfit(x, y, 3)

    def _do_step2_time_regression(self):
        x = self._simulation_data.birch_radii
        y = self._simulation_data.mean_step2_times
        self._step2_time_polyfit = np.polyfit(x, y, 3)

    def _do_clusters_after_filter_regression(self, min=0, max=9, jump=0.1):
        x = self._simulation_data.birch_radii
        y = self._simulation_data.mean_clusters_after_filter
        mask = np.where(x < 9)[0]
        x = x[mask]
        y = y[mask]
        x = np.log(x)
        y = np.log(y)
        self._log_clusters_after_filter_polyfit = np.polyfit(x, y, 2)

    def _do_clusters_after_filter_regression_2(self, min=0, max=9, jump=0.1):
        x = self._simulation_data.birch_radii
        y = np.log(self._simulation_data.mean_clusters_after_filter)
        mask = np.where(x < 9)[0]
        x = x[mask]
        y = y[mask]
        y = np.log(y)
        self._log_clusters_after_filter_polyfit_2 = np.polyfit(x, y, 3)

    def _do_avg_data_points_per_cluster_regresion(self):
        birch_radii = self._simulation_data.birch_radii
        y = self._simulation_data.avg_data_points_per_cluster
        log_birch_radii = np.log(birch_radii)
        log_y = np.log(y)
        radii_indices = np.where(birch_radii < 9)[0]
        log_birch_radii = log_birch_radii[radii_indices]
        log_y = log_y[radii_indices]
        slope, intercept,\
            r_value, p_value, std_err = stats.linregress(log_birch_radii, log_y)
        self._avg_data_points_per_cluster_slope = slope
        self._avg_data_points_per_cluster_intercept = intercept

    def _do_avg_data_points_per_cluster_regression_2(self):
        x = self._simulation_data.birch_radii
        y = self._simulation_data.avg_data_points_per_cluster
        mask = np.where(x < 5)[0]
        x = x[mask]
        y = y[mask]
        self._avg_data_points_per_cluster_polyfit_2 = np.polyfit(x, y, 2)

    def _do_log_fetched_data_regression(self):
        x = np.log(self._simulation_data.birch_radii)
        y = np.log(self._simulation_data.avg_fetched_data_points)
        self._log_fetched_data_polyfit = np.polyfit(x, y, 3)

    def _log_fetched_data_regression(self, min=0, max=9, jump=0.1):
        x = np.log(np.arange(min, max + jump, jump))
        y = np.poly1d(self._log_fetched_data_polyfit)(x)
        return x, y

    def _log_seek_time_regression(self, min=0, max=9, jump=0.1):
        x = np.log(np.arange(min, max + jump, jump))
        y = np.poly1d(self._log_seek_time_polyfit)(x)
        return x, y

    def _seek_time_regression(self, min=0, max=9, jump=0.1):
        x, y = self._log_seek_time_regression(min, max, jump)
        return np.exp(x), np.exp(y)

    def _transfer_time_regression(self, min=0, max=9, jump=0.1):
        x = np.arange(min, max + jump, jump)
        y = np.poly1d(self._transfer_time_polyfit)(x)
        return x, y

    def _step2_time_regression(self, min=0, max=9, jump=0.1):
        x = np.arange(min, max + jump, jump)
        y = np.poly1d(self._step2_time_polyfit)(x)
        return x, y

    def _step1_vs_n_clusters_regression(self):
        x = self._simulation_data.n_clusters
        y = np.poly1d(self.step1_vs_n_clusters_polyfit)(x)
        return x, y

    def _step1_vs_radius_regression(self, min=0, max=9, jump=0.1):
        x, x_inv, h = self._n_clusters_regression(min, max, jump)
        y = self.step1_vs_n_clusters_polyfit[0] * h
        return x, y

    def _step2_vs_fetched_data_regression(self):
        x = self._simulation_data.avg_fetched_data_points
        min = np.min(x)
        max = np.max(x)
        x = np.arange(min, max, (max - min)/100)
        y = np.poly1d(self.step2_vs_fetched_data_polyfit)(x)
        return x, y

    def _step2_vs_radius_regression(self, min=0, max=9, jump=0.1):
        log_x, log_h = self._log_fetched_data_regression()
        x, h = np.exp(log_x), np.exp(log_h)
        y = self.step2_vs_fetched_data_polyfit[0] * h
        print self.step2_vs_fetched_data_polyfit
        return x, y

    def _fetching_time_vs_radius_regression(self, min=0, max=9, jump=0.1):
        x = np.arange(min, max + jump, jump)
        x1, st = self._seek_time_vs_radius_regression(min, max, jump)
        x2, tt = self._transfer_time_vs_radius_regression(min, max, jump)
        y = st + tt
        return x, y

    def _seek_time_vs_radius_regression(self, min=0, max=9, jump=0.1):
        x = np.arange(min, max + jump, jump)
        log_x, log_nc = self._log_clusters_after_filter_regression(min, max, jump)
        nc = np.exp(log_nc)
        seek_time_per_cluster = 15.62 * 10**-3
        return x, nc*seek_time_per_cluster

    def _transfer_time_vs_radius_regression(self, min=0, max=9, jump=0.1):
        x = np.arange(min, max + jump, jump)
        log_nl, log_nl = self._log_fetched_data_regression(min, max, jump)
        nl = np.exp(log_nl)
        transfer_rate = 91.70 * 10**6
        data_point_size = 10*8 * 5
        tt_per_data_point = data_point_size / transfer_rate
        y = tt_per_data_point*nl
        return x, y

    def _total_time_regression(self, min=0, max=9, jump=0.1):
        x = np.arange(min, max + jump, jump)
        x_st, y_st = np.exp(self._log_seek_time_regression(min, max, jump))
        x_tt, y_tt = self._transfer_time_regression(min, max, jump)
        x_s2, y_s2 = self._step2_time_regression(min, max, jump)
        y = y_st + y_tt + y_s2
        return x, y

    def _log_seek_time_per_ts_regression(self, min=0, max=9, jump=0.1):
        x, y = self._log_seek_time_per_ts_regression(min, max, jump)
        return x, y

    def _seek_time_per_ts_regression(self, min=0, max=9, jump=0.1):
        x, y = self._log_seek_time_per_ts_regression(min, max, jump)
        return np.exp(x), np.exp(y)

    def _log_clusters_after_filter_regression(self, min=0, max=9, jump=0.1):
        x = np.log(np.arange(min, max + jump, jump))
        y = np.poly1d(self._log_clusters_after_filter_polyfit)(x)
        return x, y

    def _log_clusters_after_filter_regression_2(self, min=0, max=9, jump=0.1):
        x = np.arange(min, max + jump, jump)
        y = np.poly1d(self._log_clusters_after_filter_polyfit_2)(x)
        return x, y

    def _clusters_after_filter_regression(self, min=0, max=9, jump=0.1):
        x, y = self._log_clusters_after_filter_regression(min, max, jump)
        return np.exp(x), np.exp(y)

    def _log_avg_data_points_per_cluster_regression(self, min=0, max=9, jump=0.1):
        x = np.log(np.arange(min, max + jump, jump))
        y = self._avg_data_points_per_cluster_slope*x + self._avg_data_points_per_cluster_intercept
        return x, y

    def _avg_data_points_per_cluster_regression_2(self, min=0, max=9, jump=0.1):
        x = np.arange(min, max + jump, jump)
        y = np.poly1d(self._avg_data_points_per_cluster_polyfit_2)(x)
        return x, y

    def _log_avg_fetched_data_small_radii_regression(self, min=0, max=1, jump=0.1):
        x = np.log(np.arange(min, max + jump, jump))
        y = self._avg_fetched_data_small_radii_slope*x + self._avg_fetched_data_small_radii_intercept
        return x, y

    def _log_avg_fetched_data_large_radii_regression(self, min=1, max=9, jump=0.1):
        x = np.log(np.arange(min, max + jump, jump))
        y = self._avg_fetched_data_large_radii_slope*x + self._avg_fetched_data_large_radii_intercept
        return x, y

    def _log_avg_fetched_data_regression(self, min=0, max=9, jump=0.1):
        x1, y1 = self._log_avg_fetched_data_small_radii_regression(min, 1 - jump, jump)
        x2, y2 = self._log_avg_fetched_data_large_radii_regression(1, max, jump)
        x = np.hstack((x1, x2))
        y = np.hstack((y1, y2))
        return x, y

    def _n_clusters_regression(self, min=0, max=9, jump=0.1):
        x = np.arange(min, max + jump, jump)
        x_rec = np.exp(-1*np.log(x))
        y = np.poly1d(self._clusters_polyfit)(x_rec)
        return x, x_rec, y



    def save_pickle(self):
        with open('{0}.pkl'.format(self.name, 'regressions'), 'wb') as f:
            pickle.dump(self, f, protocol=2)

    def plot(self, save=False):
        plt.style.use('bmh')
        if save:
            self.save_pickle()
            if not os.path.exists(self.name):
                os.makedirs(self.name)
        f_a, ((a0, a1), (a2, a3)) = plt.subplots(2, 2)
        f_b, ((b0, b1), (b2, b3)) = plt.subplots(2, 2)
        f_c, ((c0, c1, c2), (c3, c4, c5)) = plt.subplots(2,3)
        self._plot_step1_time(c0, c3)
        self._plot_fetching_time(c2)
        self._plot_step2_time(c1, c4)
#        self._plot_seek_time_per_ts(a5, b5)

        self._plot_clusters(a0, b0)
        self._plot_clusters_after_filter(a1, b1)
        self._plot_avg_data_points_per_cluster(a2, b2)
        self._plot_avg_fetched_data(a3, b3)


        #self._plot_avg_data_points_per_cluster_after_first_pass(a5, b5)
        f_a.suptitle("Simulation {0}\ntargets: {1},   data points: {2}\n"
                   "Regressions".format(self._simulation_data.name, self._simulation_data.n_targets,
                                self._simulation_data._data_points_count), fontsize=16)
        f_b.suptitle("Simulation {0}\ntargets: {1},   data points: {2}\n"
                   "Regressions in log-log space".format(self._simulation_data.name, self._simulation_data.n_targets,
                                self._simulation_data._data_points_count), fontsize=16)
        f_c.suptitle("Simulation {0}\ntargets: {1},   data points: {2}\n"
                   "Regressions".format(self._simulation_data.name, self._simulation_data.n_targets,
                                self._simulation_data._data_points_count), fontsize=16)
        f_a.set_size_inches(18.5, 10.5)
        f_b.set_size_inches(18.5, 10.5)
        f_c.set_size_inches(18.5, 10.5)
        if save:
            plt.figure(f_a.number)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.savefig(os.path.join(self.name, 'regressions.jpg'))
            plt.figure(f_b.number)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.savefig(os.path.join(self.name, 'regressions modified space.jpg'))
            plt.figure(f_c.number)
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            plt.savefig(os.path.join(self.name, 'regressions execution times.jpg'))
        else:
            plt.figure(f_a.number)
            plt.show()
            plt.figure(f_b.number)
            plt.show()

    def _plot_step2_time(self, axis_r, axis_nc):
        nd_data, r_data, y_data = self._simulation_data.avg_fetched_data_points, \
                                  self._simulation_data.birch_radii, self._simulation_data.mean_step2_times
        nd_reg, y_nd_reg = self._step2_vs_fetched_data_regression()
        r_reg, y_r_reg = self._step2_vs_radius_regression()
        fit = self.step2_vs_fetched_data_polyfit
        axis_nc.scatter(nd_data, y_data)
        axis_nc.plot(nd_reg, y_nd_reg, label='$y = {0}x + {1}$'.format(fit[0], fit[1]))
        axis_nc.legend(loc=4, prop={'size': 15})
        axis_nc.set_title('Step 2 vs fetched data($S2 vs\\hat{{N_L}}$)', fontsize=12)
        axis_nc.set_xlabel('$\\hat{{N_L}}$', fontsize=12)
        axis_nc.set_ylabel('Time $(seconds)$', fontsize=12)

        axis_r.scatter(r_data, y_data)
        axis_r.plot(r_reg, y_r_reg, label='$y = {0:.2}\\hat{{N_L}}(R)$'.format(fit[0]))
        axis_r.legend(prop={'size': 15})
        axis_r.set_title('Step 2 vs radius($S2 vs R$)', fontsize=12)
        axis_r.set_xlabel('$R$', fontsize=12)
        axis_r.set_ylabel('Time $(seconds)$', fontsize=12)

    def _plot_fetching_time(self, axis):
        x_reg, y_reg = self._fetching_time_vs_radius_regression()
        x_seek, y_seek = self._seek_time_vs_radius_regression()
        x_transfer, y_transfer = self._transfer_time_vs_radius_regression()
        x_data, y_data = self._simulation_data.birch_radii, \
                                self._simulation_data.mean_fetching_times
        axis.plot(x_seek, y_seek, label='$ST = \\tau_t\\hat{N_c}$')
        axis.plot(x_transfer, y_transfer, label='$TT = s_t\\hat{N_L}$')
        axis.plot(x_reg, y_reg, label='$FT = ST + TT$')
        axis.scatter(x_data, y_data)
        axis.legend(loc=4, prop={'size': 15})
        axis.set_title('Fetching Time (FT)', fontsize=12)
        axis.set_xlabel('$R$', fontsize=12)
        axis.set_ylabel('Time $(seconds)$', fontsize=12)

    def _plot_step1_time(self, axis_r, axis_nc):
        nc_data, r_data, y_data = self._simulation_data.n_clusters, \
                                  self._simulation_data.birch_radii, self._simulation_data.mean_step1_times
        nc_reg, y_nc_reg = self._step1_vs_n_clusters_regression()
        r_reg, y_r_reg = self._step1_vs_radius_regression()
        fit = self.step1_vs_n_clusters_polyfit
        axis_nc.scatter(nc_data, y_data)
        axis_nc.plot(nc_reg, y_nc_reg, label='$y = {0}x + {1}$'.format(fit[0], fit[1]))
        axis_nc.legend(loc=4, prop={'size': 15})
        axis_nc.set_title('Step 1 vs number of clusters($S1 vs N_c$)', fontsize=12)
        axis_nc.set_xlabel('$N_c$', fontsize=12)
        axis_nc.set_ylabel('Time $(seconds)$', fontsize=12)

        axis_r.scatter(r_data, y_data)
        axis_r.plot(r_reg, y_r_reg, label='$y = {0:.2}N_c(R)$'.format(fit[0]))
        axis_r.legend(prop={'size': 15})
        axis_r.set_title('Step 1 vs radius($S1 vs R$)', fontsize=12)
        axis_r.set_xlabel('$R$', fontsize=12)
        axis_r.set_ylabel('Time $(seconds)$', fontsize=12)


    def _plot_total_time(self, axis_lin, axis_log):
        x_reg, y_reg = self._total_time_regression()
        x_sim, y_sim = self._simulation_data.birch_radii, self._simulation_data.mean_step2_times
        fit = self._step2_time_polyfit
        axis_lin.plot(x_reg, y_reg, label='$T = ST + TT + S2$')
        axis_lin.scatter(x_sim, y_sim)
        axis_lin.legend(loc=4, prop={'size': 15})
        axis_lin.set_title('Total Time (T)', fontsize=12)
        axis_lin.set_xlabel('$R$', fontsize=12)
        axis_lin.set_ylabel('$seconds$', fontsize=12)
        axis_log.scatter(np.log(x_sim), np.log(y_sim))
        axis_log.legend(loc=4, prop={'size': 15})
        axis_log.set_title('Total time (T)', fontsize=12)
        axis_log.set_xlabel('$log(R)$', fontsize=12)
        axis_log.set_ylabel('$log(seconds)$', fontsize=12)

    def _plot_clusters(self, axis_lin, axis_rec):
        x_reg, x_rec_reg, y_reg = self._n_clusters_regression()
        fit = self._clusters_polyfit
        x_data, y_data = self._simulation_data.birch_radii, self._simulation_data.n_clusters
        x_data_rec = np.exp(-1*np.log(x_data))
        axis_lin.scatter(x_data, y_data)
        print x_reg
        print y_reg
        axis_lin.plot(x_reg, y_reg)
        axis_lin.legend(prop={'size': 15})
        axis_lin.set_title('# clusters($N_C$)', fontsize=12)
        axis_lin.set_xlabel('$R$', fontsize=12)
        axis_lin.set_ylabel('$N_C$', fontsize=12)
        axis_lin.plot(x_reg, y_reg,
                 label='$y = \\frac{{{0:.3}}}{{x^2}} + \\frac{{{1:.3}}}{{x}} + {2:.3}$'.format(fit[0], fit[1], fit[2]))
        axis_lin.legend(prop={'size': 15}, loc=0)
        # axis_rec.plot(x_reg, y_reg,
        #          label='$y = {0:.3}x^2 + {1:.3}x + {2:.3}$'.format(fit[0], fit[1], fit[2]))
        axis_rec.scatter(x_data_rec, y_data)
        axis_rec.plot(x_rec_reg, y_reg)
        axis_rec.legend(prop={'size': 15}, loc=0)
        axis_rec.set_title('# clusters ($N_C$)', fontsize=12)
        axis_rec.set_xlabel('$\\frac{1}{R}$', fontsize=12)
        axis_rec.set_ylabel('$N_C$', fontsize=12)

    def _plot_clusters_after_filter(self, axis_lin, axis_log):
        x_reg, y_reg = self._clusters_after_filter_regression()
        x_data, y_data = self._simulation_data.birch_radii, self._simulation_data.mean_clusters_after_filter
        log_x_reg, log_y_reg = self._log_clusters_after_filter_regression()
        log_x_data, log_y_data = np.log(x_data), np.log(y_data)
        log_fit = self._log_clusters_after_filter_polyfit
        axis_lin.plot(x_reg, y_reg,
                 label='$y = {0:.2f}x^{{{1:.2f}}}e^{{ {2:.2f}\log^2(x) }}$'.format(np.exp(log_fit[2]), log_fit[1], log_fit[0]))
        axis_lin.scatter(x_data, y_data)
        axis_lin.legend(prop={'size': 15})
        axis_lin.set_title('# clusters after first pass ($\hat{N_C}$)', fontsize=12)
        axis_lin.set_xlabel('$R$', fontsize=12)
        axis_lin.set_ylabel('$\hat{N_C}$', fontsize=12)
        axis_log.plot(log_x_reg, log_y_reg,
                 label='$y={0:.2f} + {1:.2f}x + {2:.2f}x^2$'.format(log_fit[2], log_fit[1], log_fit[0]))
        axis_log.scatter(log_x_data, log_y_data)
        axis_log.legend(prop={'size': 15}, loc=3)
        axis_log.set_title('# clusters after first pass ($\hat{N_C}$)', fontsize=12)
        axis_log.set_xlabel('$log(R)$', fontsize=12)
        axis_log.set_ylabel('$log(\hat{N_C})$', fontsize=12)

    def _plot_avg_data_points_per_cluster(self, axis_lin, axis_log):
        log_x_reg, log_y_reg = self._log_avg_data_points_per_cluster_regression()
        x_reg, y_reg = np.exp(log_x_reg), np.exp(log_y_reg)
        x_reg_2, y_reg_2 = self._avg_data_points_per_cluster_regression_2()
        x_data, y_data = self._simulation_data.birch_radii, self._simulation_data.avg_data_points_per_cluster
        fit = self._avg_data_points_per_cluster_polyfit_2
        mask = np.where(x_data < 9)[0]
        x_data = x_data[mask]
        y_data = y_data[mask]
        log_x_data, log_y_data = np.log(x_data), np.log(y_data)
        axis_lin.plot(x_reg, y_reg,
                 label='$\\mathbf{{n: }} y = {0:.2f}x^{{{1:.2f}}}$'.format(self._avg_data_points_per_cluster_coefficient, self._avg_data_points_per_cluster_exponent))
        axis_lin.plot(x_reg_2, y_reg_2,
                label='$y={0:.5f} + {1:.5f}x + {2:.5f}x^2$'.format(fit[2], fit[1], fit[0]))
        axis_lin.scatter(x_data, y_data)
        axis_lin.legend(prop={'size': 15})
        axis_lin.set_title('# light curves per cluster:\n before first pass ($n$), after first pass ($\hat{n}$)', fontsize=12)
        axis_lin.set_xlabel('$R$', fontsize=12)
        axis_lin.set_ylabel('$n$', fontsize=12)
        axis_log.plot(log_x_reg, log_y_reg,
                 label='$y = {0:.2f}x + {1:.2f}$'.format(self._avg_data_points_per_cluster_slope, self._avg_data_points_per_cluster_intercept))
        axis_log.scatter(log_x_data, log_y_data)
        axis_log.legend(prop={'size': 15})
        axis_log.set_title('# light curves per cluster ($n$)', fontsize=12)
        axis_log.set_xlabel('$log(R)$', fontsize=12)
        axis_log.set_ylabel('$log(n)$', fontsize=12)

    def _plot_avg_fetched_data(self, axis_lin, axis_log):
        x1, y1_1 = self._log_avg_data_points_per_cluster_regression()
        x1, y1_2 = self._log_clusters_after_filter_regression()
        y1 = y1_1 + y1_2
        log_x_reg, log_y_reg = self._log_fetched_data_regression()
        x_reg, y_reg = np.exp(log_x_reg), np.exp(log_y_reg)
        log_fit = self._log_fetched_data_polyfit
        axis_lin.plot(x_reg, y_reg,
                 label='$y = {0:.2f}x^{{{1:.2f}}}e^{{ {2:.2f}\log^2(x) + {3:.2f}\log^3(x)}}$'.format(np.exp(log_fit[3]), log_fit[2], log_fit[1], log_fit[0]))
        axis_lin.plot(np.exp(x1), np.exp(y1),
         label='$y = n\hat{N_c}$')
        birch_radii = self._simulation_data.birch_radii
        avg_fetched_data = self._simulation_data.avg_fetched_data_points
        # indices = np.where(birch_radii < 9)
        # birch_radii = birch_radii[indices]
        # avg_d_p_c = avg_d_p_c[indices]
        axis_lin.scatter(birch_radii, avg_fetched_data)
        axis_lin.legend(loc=4, prop={'size': 15})
        axis_lin.set_title('# fetched data points in second pass ($\hat{N_L}$)', fontsize=12)
        axis_lin.set_xlabel('$R$', fontsize=12)
        axis_lin.set_ylabel('# data points', fontsize=12)
        axis_log.plot(log_x_reg, log_y_reg,
                 label='$y={0:.2f} + {1:.2f}x + {2:.2f}x^2 + + {3:.2f}x^3$'.format(log_fit[3], log_fit[2], log_fit[1], log_fit[0]))
        axis_log.scatter(np.log(self._simulation_data.birch_radii), np.log(self._simulation_data.avg_fetched_data_points))
        axis_log.plot(x1, y1,
         label='$y = \log(n)  + \log(\hat{N_c})$')
        axis_log.legend(loc=4, prop={'size': 15})
        axis_log.set_title('# fetched data points in second pass ($\hat{N_L}$)', fontsize=12)
        axis_log.set_xlabel('$log(R)$', fontsize=12)
        axis_log.set_ylabel('log(# data points)', fontsize=12)


class SimulationData(object):

    def __init__(self, name, step1_calc_times, step2_calc_times,
                 fetching_times, n_data_after_filter,
                 n_visited_clusters_list, clusters_count,
                 clustering_dbs_names, data_points_count, birch_radii, clusters_radii):
        self.name = name
        self._step1_calc_times = step1_calc_times
        self._step2_calc_times = step2_calc_times
        self._fetching_times = fetching_times
        self._n_data_after_filter = n_data_after_filter
        self._n_visited_clusters_list = n_visited_clusters_list
        self._clusters_count = clusters_count
        self._clustering_dbs_names = clustering_dbs_names
        self._data_points_count = data_points_count
        self._birch_radii = birch_radii
        self._clusters_radii = clusters_radii
        self._regression_data = None

    @property
    def n_dbs(self):
        return len(self._step1_calc_times)

    @property
    def n_clusters(self):
        return np.array([len(clusters) for clusters in self._clusters_count])

    @property
    def mean_seek_times(self):
        return np.array([np.mean(seek_times) for seek_times in self._seek_times])

    @property
    def mean_fetching_times(self):
        return np.array([np.mean(times) for times in self._fetching_times])

    @property
    def mean_step1_times(self):
        return np.array([np.mean(times) for times in self._step1_calc_times])

    @property
    def mean_step2_times(self):
        return np.array([np.mean(times) for times in self._step2_calc_times])

    @property
    def total_times(self):
        result = []
        for step1, step2, seek, transfer in zip(
                self._step1_calc_times, self._step2_calc_times,
                self._seek_times, self._transfer_times):
            total = np.array(step1) + np.array(step2) + np.array(seek) + np.array(transfer)
            result.append(total.tolist())
        return result

    @property
    def mean_total_times(self):
        return np.array([np.mean(times) for times in self._total_times])

    @property
    def mean_clusters_after_filter(self):
        return np.array([np.mean(fetched_clusters) for fetched_clusters in self._n_visited_clusters_list])

    @property
    def birch_radii(self):
        return np.array(self._birch_radii)

    @property
    def avg_data_points_per_cluster(self):
        return np.array([np.mean(cluster_count) for cluster_count in self._clusters_count])

    @property
    def avg_data_points_per_cluster_after_first_pass(self):
        return np.array([np.mean(cluster_count) for cluster_count in self._clusters_count])
        return np.array([np.mean(cluster_count) for cluster_count in self._clusters_count])

    @property
    def avg_fetched_data_points(self):
        return np.array([np.mean(data_points_count) for data_points_count in self._n_data_after_filter])

    @property
    def _total_times(self):
        total_times = []
        for step1, step2, fetch, in zip(
                self._step1_calc_times, self._step2_calc_times,
                self._fetching_times):
            step1_arr = np.array(step1)
            step2_arr = np.array(step2)
            fetch_arr = np.array(fetch)
            total_arr = step1_arr + step2_arr + fetch_arr
            total_times.append(total_arr.tolist())
        return total_times

    @property
    def n_targets(self):
        return len(self._step1_calc_times[0])

    def plot_all(self, n_rows, n_bins=50, save=False):
        if save:
            self.save_pickle()
            if not os.path.exists(self.name):
                os.makedirs(self.name)
                os.makedirs(os.path.join(self.name, 'histograms'))
        self._plot_step1_calc(n_rows, n_bins, save)
        self.plot_step2_calc(n_rows, n_bins, save)
        self.plot_fetching_time(n_rows, n_bins, save)
        self.plot_n_data_after_filter(n_rows, n_bins, save)
        self.plot_n_visited_clusters_list(n_rows, n_bins, save)
        self.plot_n_clusters_count(n_rows, n_bins, save)
        self.plot_times(save)
        self.plot_fetched_data(save)
        self.plot_data_distribution(save)
        self.plot_total_time(save)
        #self.plot_log_seek_time(save)

    def _plot_step1_calc(self, n_rows, n_bins, save=False):
        self._plot_hist_panel(n_rows, n_bins, save, self._step1_calc_times, "Step 1 calculations")

    def plot_step2_calc(self, n_rows, n_bins, save=False):
        self._plot_hist_panel(n_rows, n_bins, save, self._step2_calc_times, "Step 2 calculations")

    def plot_fetching_time(self, n_rows, n_bins, save=False):
        self._plot_hist_panel(n_rows, n_bins, save, self._fetching_times, "Fetching time")

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
        sorted_values = []
        sorted_radii = []
        for (radius, v) in sorted(zip(self._birch_radii, values)):
            sorted_radii.append(radius)
            sorted_values.append(v)
        for ax, v, radius in zip(axes_flat, sorted_values, sorted_radii):
            n_bins = min(n_bins, len(v))
            r = None
            if len(v) == 1:
                r = (v[0] - 1, v[0] + 1)
            ax.hist(v, bins=n_bins, range=r)
            ax.set_xlabel("Birch radius = {0}".format(radius))
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
            plt.savefig(os.path.join(self.name, 'histograms' ,'{0}.jpg'.format(plot_name)))
        else:
            plt.show()

    def plot_log_seek_time(self, save=False):
        f, (ax0, ax1) = plt.subplots(2)
        birch_radii = np.array(self._birch_radii)
        mean_seek_times = [np.mean(seek_times) for seek_times in self._seek_times]
        log_birch_radii = np.log(self._birch_radii)
        log_seek_times = np.log(mean_seek_times)
        ax0.scatter(self._birch_radii, mean_seek_times)
        ax0.set_xlabel("Birch radius")
        ax0.set_ylabel("Time (seconds)")
        ax0.set_title('Database mean seek times')
        mean_seek_times = np.array([np.mean(seek_times) for seek_times in self._seek_times])
        ax1.scatter(np.log(self._birch_radii), np.log(mean_seek_times))
        ax1.set_xlabel("Birch radius (log space)")
        ax1.set_ylabel("Time (log space)")
        ax1.set_title('Database mean seek times (log-log plot)')
        small_radii_indices = np.where(birch_radii <= 1)[0]
        large_radii_indices = np.where(np.logical_and(birch_radii >= 1, birch_radii < 9))[0]
        small_log_birch_radii = log_birch_radii[small_radii_indices]
        large_log_birch_radii = log_birch_radii[large_radii_indices]
        small_radii_log_seek = log_seek_times[small_radii_indices]
        large_radii_log_seek = log_seek_times[large_radii_indices]
        small_radii_slope, small_radii_intercept,\
            r_value, p_value, std_err = stats.linregress(small_log_birch_radii, small_radii_log_seek)
        large_radii_slope, large_radii_intercept,\
            r_value, p_value, std_err = stats.linregress(large_log_birch_radii, large_radii_log_seek)
        small_radii_regression_x = np.log(np.arange(0, 1.1, 0.1))
        large_radii_regression_x = np.log(np.arange(1, 9, 0.1))
        small_radii_regression_y = small_radii_slope*small_radii_regression_x + small_radii_intercept
        large_radii_regression_y = large_radii_slope*large_radii_regression_x + large_radii_intercept
        small_radii_exponent = small_radii_slope
        large_radii_exponent = large_radii_slope
        small_radii_coefficient = np.exp(small_radii_intercept)
        large_radii_coefficient = np.exp(large_radii_intercept)
        ax0.plot(np.exp(small_radii_regression_x), np.exp(small_radii_regression_y),
                 label='$y = {0}x^{{{1}}}$'.format(small_radii_coefficient, small_radii_exponent))
        ax0.plot(np.exp(large_radii_regression_x), np.exp(large_radii_regression_y),
                 label='$y = {0}x^{{{1}}}$'.format(large_radii_coefficient, large_radii_exponent))
        ax0.legend()
        ax1.plot(small_radii_regression_x, small_radii_regression_y,
                 label='$y = {0}x + {1}$'.format(small_radii_slope, small_radii_intercept))
        ax1.plot(large_radii_regression_x, large_radii_regression_y,
                 label='$y = {0}x + {1}$'.format(large_radii_slope, large_radii_intercept))
        ax1.legend()
        f.suptitle("Simulation {0}\ntargets: {1},   data points: {2}\n"
                   "Mean seek times".format(self.name, self.n_targets,
                                self._data_points_count), fontsize=16)
        f.set_size_inches(18.5, 10.5)
        plt.subplots_adjust(top=0.85)

        if save:
            plt.savefig(os.path.join(self.name, 'log_seek_time.jpg'))
        else:
            plt.show()

    def plot_times(self, save=False):
        f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
        ax0.boxplot(self._step1_calc_times, positions=self._birch_radii)
        ax0.plot(self._birch_radii, self.mean_step1_times, 'o', color='green')
        ax0.set_xticks(self._birch_radii)
        ax0.set_xticklabels(self._birch_radii, rotation=45)
        ax0.set_ylabel('Time (seconds)')
        ax0.set_xlabel('Birch radius')
        ax0.set_title('Step 1 calculations')
        ax1.boxplot(self._step2_calc_times, positions=self._birch_radii)
        ax1.plot(self._birch_radii, self.mean_step2_times, 'o', color='green')
        ax1.set_xticks(self._birch_radii)
        ax1.set_xticklabels(self._birch_radii, rotation=45)
        ax1.set_ylabel('Time (seconds)')
        ax1.set_xlabel('Birch radius')
        ax1.set_title('Step 2 calculations')
        ax2.boxplot(self._fetching_times, positions=self._birch_radii)
        ax2.plot(self._birch_radii, self.mean_fetching_times, 'o', color='green')
        ax2.set_xticks(self._birch_radii)
        ax2.set_xticklabels(self._birch_radii, rotation=45)
        ax2.set_xlabel('Birch radius')
        ax2.set_ylabel('Time (seconds)')
        ax2.set_title('Fetching time')
        ax3.boxplot(self._total_times, positions=self._birch_radii)
        ax3.plot(self._birch_radii, self.mean_total_times, 'o', color='green')
        ax3.set_xticks(self._birch_radii)
        ax3.set_xticklabels(self._birch_radii, rotation=45)
        ax3.set_ylabel('Time (seconds)')
        ax3.set_xlabel('Birch radius')
        ax3.set_title('Total time calculations')
        f.suptitle("Simulation {0}\ntargets: {1},   data points: {2}\n"
                   "Execution times".format(self.name, self.n_targets,
                                self._data_points_count), fontsize=16)
        f.set_size_inches(18.5, 10.5)
        plt.subplots_adjust(top=0.85)

        if save:
            plt.savefig(os.path.join(self.name, 'execution_times.jpg'))
        else:
            plt.show()

    def plot_total_time(self, save=False):
        f, ax = plt.subplots(1)
        print self._total_times

        f.suptitle("Simulation {0}\ntargets: {1},   data points: {2}\n"
                   "Execution times".format(self.name, self.n_targets,
                                self._data_points_count), fontsize=16)
        f.set_size_inches(18.5, 10.5)
        plt.subplots_adjust(top=0.85)
        if save:
            plt.savefig(os.path.join(self.name, 'total_time.jpg'))
        else:
            plt.show()

    def plot_fetched_data(self, save=False):
        f, (ax0, ax1) = plt.subplots(1, 2)
        mean_n_visited_clusters = self.mean_clusters_after_filter
        ax0.boxplot(self._n_visited_clusters_list, positions=self._birch_radii)
        ax0.set_xticks(self._birch_radii)
        ax0.set_xticklabels(self._birch_radii, rotation=45)
        ax0.set_xlabel('Birch radius')
        ax0.set_ylabel('Count')
        ax0.set_title('Clusters fetched')
        ax0.plot(self._birch_radii, mean_n_visited_clusters, 'o', color='green')
        ax1.boxplot(self._n_data_after_filter, positions=self._birch_radii)
        ax1.set_xticks(self._birch_radii)
        ax1.set_xticklabels(self._birch_radii, rotation=45)
        ax1.set_xlabel('Birch radius')
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
        n_clusters = []
        birch_radii = []
        mean_clusters_count = []
        clusters_count = []
        clusters_radii = []
        for nc, r, cc, mcc, cr in zip(self.n_clusters, self._birch_radii, self._clusters_count,
                          self.avg_data_points_per_cluster, self._clusters_radii):
            if r < 9:
                birch_radii.append(r)
                n_clusters.append(nc)
                clusters_count.append(cc)
                mean_clusters_count.append(mcc)
                clusters_radii.append(cr)
        f, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
        ax0.boxplot(clusters_count, positions=birch_radii)
        ax0.set_xticklabels(birch_radii, rotation=45)
        ax0.set_xlabel("Birch radius")
        ax0.set_ylabel('Data points per cluster')
        ax0.plot(birch_radii, mean_clusters_count, 'o', color='green')

        ax1.scatter(birch_radii, n_clusters, marker=".")
        ax1.set_xlabel("Birch radius")
        ax1.set_ylabel('Number of clusters')

        ax2.boxplot(clusters_radii, positions=birch_radii)
        ax2.set_xticklabels(birch_radii, rotation=45)
        ax2.set_xlabel("Birch radius")
        ax2.set_ylabel('Clusters radii')

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
        self.simulation_data = None

    @property
    def count(self):
        return self._clustering_dbs[0].data_points_count


    @property
    def radii(self):
        return [clustering_db.metadata['radius'] for clustering_db in self._clustering_dbs]

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
        fetching_times = []
        step2_calc_times = []
        n_data_after_filters = []
        n_visited_clusters_list = []
        for time_series in random_time_series:
            retrieved_time_series, retrieved_distances, \
                step1_calc_time, step2_calc_time, fetching_time, n_data_after_filter,\
                n_visited_clusters = self.feature_space_query(time_series.feature_dict, k_neighbors, indexing_model)
            step1_calc_times.append(step1_calc_time)
            step2_calc_times.append(step2_calc_time)
            fetching_times.append(fetching_time)
            n_data_after_filters.append(n_data_after_filter)
            n_visited_clusters_list.append(n_visited_clusters)
        clusters_count = clustering_db.counts.tolist()
        clusters_radii = clustering_db.radii.tolist()
        return step1_calc_times, step2_calc_times,\
               fetching_times,\
               n_data_after_filters, n_visited_clusters_list, clusters_count, clusters_radii

    def do_all_simulations(self, n_time_series, k_neighbors):
        all_step1_calc_times = []
        all_step2_calc_times = []
        all_fetching_times = []
        all_n_data_after_filters = []
        all_n_visited_clusters_list = []
        all_clusters_count = []
        all_clusters_radii = []
        for indexing_model, clustering_db in zip(self._indexing_models, self._clustering_dbs):
            print("Running simulations over {0} database".format(clustering_db.db_name))
            step1_calc_times, step2_calc_times,\
            fetching_times, n_data_after_filters,\
            n_visited_clusters_list, clusters_count, clusters_radii = self._do_one_simulation(
                n_time_series, k_neighbors, indexing_model, clustering_db)
            all_step1_calc_times.append(step1_calc_times)
            all_step2_calc_times.append(step2_calc_times)
            all_fetching_times.append(fetching_times)
            all_n_data_after_filters.append(n_data_after_filters)
            all_n_visited_clusters_list.append(n_visited_clusters_list)
            all_clusters_count.append(clusters_count)
            all_clusters_radii.append(clusters_radii)
        self.simulation_data = SimulationData(self.name, all_step1_calc_times,
                                  all_step2_calc_times, all_fetching_times, all_n_data_after_filters,
                                  all_n_visited_clusters_list, all_clusters_count,
                                  [cluster_db.db_name for cluster_db in self._clustering_dbs], self.count, self.radii, all_clusters_radii)


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
    clustering_dbs_indices = range(9) + [10, 11, 12, 13, 14, 15, 16, 17]
    simulations_interface = BirchMultiDatabaseSimulationsInterface('Macho Field 1 v2',
                                                                   data_model_interface, 0, clustering_dbs_indices, 0)
    simulations_interface.do_all_simulations(100, 1)
    simulations_interface.simulation_data.save_pickle()


def fetch_and_plot_simulation(file_name, n_rows, n_bins, save):
    with open(file_name, 'r') as f:
        simulation_data = pickle.load(f)
        print simulation_data._birch_radii
        simulation_data.plot_all(n_rows, n_bins, save)
        regression_data = RegressionData(simulation_data)
        regression_data.plot(save)

if __name__ == '__main__':
    #run_t1_t9_simulation()
    fetch_and_plot_simulation('Macho Field 1 v2.pkl', 2, 50, True)



