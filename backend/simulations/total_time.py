import matplotlib.pyplot as plt
import numpy as np


class ExecutionTimes(object):

    def __init__(self, time_per_light_curve_step1, time_per_light_curve_step2,
                 transfer_rate, seek_time, dimensionality, scalar_size, metadata_size,
                 n_clusters_function, n_cluster_after_pass_function, n_lc_after_pass_function,
                 n_lc_per_cluster_function, n_lc, gpu_memory):
        self.time_per_light_curve_step1 = time_per_light_curve_step1
        self.time_per_light_curve_step2 = time_per_light_curve_step2
        self.transfer_rate = transfer_rate
        self.hardware_seek_time = seek_time
        self.dimensionality = dimensionality
        self.scalar_size = scalar_size
        self.metadata_size = metadata_size
        self.n_clusters_function = n_clusters_function
        self.n_cluster_after_pass_function = n_cluster_after_pass_function
        self.n_lc_after_pass_function = n_lc_after_pass_function
        self.n_lc_per_cluster_function = n_lc_per_cluster_function
        self.n_lc = n_lc
        self.gpu_memory = gpu_memory

    @property
    def light_curve_size(self):
        return self.dimensionality*self.scalar_size

    @property
    def max_number_of_clusters(self):
        m = self.gpu_memory/(self.dimensionality*4)
        return self.gpu_memory/(self.dimensionality*4)

    def max_number_of_clusters_radius(self, x):
        n_clusters = self.n_clusters_function(x)
        index = np.argmin(n_clusters > self.max_number_of_clusters)
        return x[index]

    def step1_time(self, radius):
        return self.time_per_light_curve_step1*self.n_clusters_function(radius)

    def seek_time(self, radius):
        return self.hardware_seek_time*self.n_cluster_after_pass_function(radius)

    def transfer_time(self, radius):
        return 1000*(self.n_cluster_after_pass_function(radius)*self.metadata_size +
                self.n_lc_after_pass_function(radius)*self.light_curve_size)/self.transfer_rate

    def step2_time(self, radius):
        return 1000*self.time_per_light_curve_step2 * self.n_lc_after_pass_function(radius)

    def total_time(self, radius):
        return self.step1_time(radius) + self.seek_time(radius) + self.transfer_time(radius) + self.step2_time(radius)

    def plot(self, logspace_x, logspace_y, min=0, max=9, jump=0.001):
        plt.style.use('bmh')
        x = np.arange(min, max, jump)
        step1 = self.step1_time(x)
        seek = self.seek_time(x)
        transfer = self.transfer_time(x)
        step2 = self.step2_time(x)
        total = self.total_time(x)
        gpu_limit_x = self.max_number_of_clusters_radius(x)
        gpu_limit_y = self.max_number_of_clusters
        print gpu_limit_y
        n_clusters = self.n_clusters_function(x)
        print n_clusters
        start_str_x = ''
        end_str_x = ''
        start_str_y = ''
        end_str_y = ''
        if logspace_x:
            x = np.log(x)
            gpu_limit_x = np.log(gpu_limit_x)
            start_str_x = 'log('
            end_str_x = ')'
        if logspace_y:
            step1 = np.log(step1)
            seek = np.log(seek)
            transfer = np.log(transfer)
            step2 = np.log(step2)
            total = np.log(total)
            gpu_limit_y = np.log(gpu_limit_y)
            n_clusters = np.log(n_clusters)
            start_str_y = 'log('
            end_str_y = ')'
        f, (ax1, ax2) = plt.subplots(2, sharex=True)
        f.suptitle('Predicted execution times for {0} lightcurves dataset'.format(self.n_lc))
        ax1.plot(x, n_clusters)
        ax1.set_title('Number of clusters', fontsize=12)
        ax2.set_xlabel('${0}R{1}$'.format(start_str_x, end_str_x), fontsize=12)
        ax1.set_ylabel('${0}\# clusters{1}$'.format(start_str_y, end_str_y), fontsize=12)
        ax2.plot(x, total, label='total')
        ax2.plot(x, step1, '--', label='step 1')
        ax2.plot(x, seek, '--', label='seek')
        ax2.plot(x, transfer, '--', label='transfer')
        ax2.plot(x, step2, '--', label='step 2')
        ax2.legend(prop={'size': 15})
        ax2.set_title('Execution times', fontsize=12)
        ax2.set_xlabel('${0}R{1}$'.format(start_str_x, end_str_x), fontsize=12)
        ax2.set_ylabel('${0}seconds{1}$'.format(start_str_y, end_str_y), fontsize=12)
        # ax3 = ax1.twiny()
        # ax3.set_xlim(ax1.get_xlim())
        # x_ticks = np.arange(min, max, 0.5)
        # ax3.set_xticks(x_ticks)
        # ax3.set_xticklabels(self.n_clusters_function(x_ticks), rotation='vertical')
        plt.ylim([-3, 3])
        ax1.axvline(x=gpu_limit_x)
        ax2.axvline(x=gpu_limit_x)
        ax1.axhline(y=gpu_limit_y)
        plt.subplots_adjust(top=0.85)
        plt.show()


    def plot_nc(self):
        x = np.arange(0, 9, 0.1)
        y = self.n_clusters_function(x)
        plt.plot(x, y)
        plt.show()

if __name__ == '__main__':
    time_per_light_curve_step1 = 1.3e-7
    time_per_light_curve_step2 = 3.9e-7
    transfer_rate = 91.70 * 10**6
    seek_time = 15.62 * 10**-4
    dimensionality = 50
    scalar_size = 24
    metadata_size = 160
    number_of_lc = 436865
    gpu_memory = 12*10**6
    n_clusters_function = lambda x: 6.31e3/x**2 + 1.15e4/x - 4.3e3
    n_cluster_after_pass_function = lambda x: np.exp(-0.92*np.log(x)**2+0.33*np.log(x)+6.52)
    n_lc_after_pass_function = lambda x: np.exp(0.12*np.log(x)**3 - 0.94*np.log(x)**2 + 1.99*np.log(x) + 11.82)
    n_lc_per_cluster = lambda x: 56.10*x**2 - 38.37*x + 12.08
    times = ExecutionTimes(time_per_light_curve_step1, time_per_light_curve_step2,
                           transfer_rate, seek_time, dimensionality, scalar_size, metadata_size,
                           n_clusters_function, n_cluster_after_pass_function,
                           n_lc_after_pass_function, n_lc_per_cluster, number_of_lc, gpu_memory)
    times.plot(False, False)