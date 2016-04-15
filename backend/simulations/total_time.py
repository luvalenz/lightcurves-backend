import matplotlib.pyplot as plt
import numpy as np


class ExecutionTimes(object):

    def __init__(self, step1_time_per_light_curve, seek_time_per_light_curve,
                 transfer_rate, lc_size, step2_time_per_light_curve,
                 n_clusters_function, n_cluster_after_pass_function, n_lc_per_cluster_after_pass_function):
        self.step1_time_per_light_curve = step1_time_per_light_curve
        self.seek_time_per_light_curve = seek_time_per_light_curve
        self.transfer_rate = transfer_rate
        self.light_curve_size = lc_size
        self.step2_time_per_light_curve = step2_time_per_light_curve
        self.n_clusters_function = n_clusters_function
        self.n_cluster_after_pass_function = n_cluster_after_pass_function
        self.n_lc_per_cluster_after_pass_function = n_lc_per_cluster_after_pass_function

    @property
    def transfer_time_per_light_curve(self):
        return self.transfer_rate*self.light_curve_size

    def step1_time(self, radius):
        return self.step1_time_per_light_curve*self.n_clusters_function(radius)

    def seek_time(self, radius):
        return self.seek_time_per_light_curve*self.n_cluster_after_pass_function(radius)

    def transfer_time(self, radius):
        return self.transfer_time_per_light_curve*self.n_lc_per_cluster_after_pass_function(radius)*self.n_cluster_after_pass_function(radius)

    def step2_time(self, radius):
        return self.step2_time_per_light_curve*self.n_lc_per_cluster_after_pass_function(radius)*self.n_cluster_after_pass_function(radius)

    def plot(self, min=0, max=9, jump=0.1):
        x = np.arange(min, max, jump)
        step1 = self.step1_time(x)
        seek = self.seek_time(x)
        transfer = self.transfer_time(x)
        step2 = self.step2_time(x)
        total = step1 + seek + transfer + step2
        f, ax = plt.subplots(1)
        ax.plot(x, step1, label='step 1')
        ax.plot(x, seek, label='seek')
        ax.plot(x, transfer, label='transfer')
        ax.plot(x, transfer, label='step 2')
        ax.plot(x, total, label='total')
        ax.legend(prop={'size': 15})
        ax.set_title('Execution times', fontsize=12)
        ax.set_xlabel('$R$', fontsize=12)
        ax.set_ylabel('$seconds$', fontsize=12)
        f.set_size_inches(18.5, 10.5)
        plt.subplots_adjust(top=0.85)
        plt.show()

if __name__ == '__main__':
    step1_time_per_light_curve = 1
    seek_time_per_light_curve = 2
    transfer_rate = 3
    light_curve_size = 4
    step2_time_per_light_curve = 5
    n_clusters_function = lambda x: 4*x+1
    n_cluster_after_pass_function = lambda x: 5*x+3
    n_lc_per_cluster_after_pass_function = lambda x: 4*x**2
    times = ExecutionTimes(step1_time_per_light_curve, seek_time_per_light_curve, transfer_rate, light_curve_size,
                           step2_time_per_light_curve, n_clusters_function, n_cluster_after_pass_function, n_lc_per_cluster_after_pass_function)
    times.plot()