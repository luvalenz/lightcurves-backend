import matplotlib.pyplot as plt
import numpy as np


class ExecutionTimes(object):

    def __init__(self, cpu_speed, seek_cluster,
                 transfer_rate, lc_length, gpu_speed,
                 n_clusters_function, n_cluster_after_pass_function,
                 n_lc_per_cluster_after_pass_function,
                 paralelization_factor, operations_per_comparison, data_type_size):
        self.cpu_speed = cpu_speed
        self.seek_time_per_cluster = seek_time_per_cluster
        self.transfer_rate = transfer_rate
        self.light_curve_length = lc_length
        self.gpu_speed = gpu_speed
        self.n_clusters_function = n_clusters_function
        self.n_cluster_after_pass_function = n_cluster_after_pass_function
        self.n_lc_per_cluster_after_pass_function = n_lc_per_cluster_after_pass_function
        self.paralelization_factor = paralelization_factor
        self.operations_per_comparison = operations_per_comparison
        self.data_type_size = data_type_size

    @property
    def time_per_light_curve_cpu(self):
        return self.light_curve_size/self.cpu_speed


    @property
    def time_per_light_curve_gpu(self):
        return self.light_curve_size/self.gpu_speed

    @property
    def light_curve_size(self):
        a = self.light_curve_length*self.data_type_size
        print a
        return a

    @property
    def transfer_time_per_light_curve(self):
        a=  self.light_curve_size/self.transfer_rate
        print a
        return a

    def step1_time(self, radius):
        return self.time_per_light_curve_cpu*self.n_clusters_function(radius)

    def step1_time_gpu(self, radius):
        return self.time_per_light_curve_gpu*self.n_clusters_function(radius)/paralelization_factor

    def seek_time(self, radius):
        return self.seek_time_per_cluster*self.n_cluster_after_pass_function(radius)

    def transfer_time(self, radius):
        return self.transfer_time_per_light_curve*self.n_lc_per_cluster_after_pass_function(radius)*self.n_cluster_after_pass_function(radius)

    def step2_time(self, radius):
        return self.time_per_light_curve_cpu*self.n_lc_per_cluster_after_pass_function(radius)*self.n_cluster_after_pass_function(radius)

    def plot(self, min=0, max=9, jump=0.1):
        plt.style.use('bmh')
        x = np.arange(min, max, jump)
        step1 = self.step1_time(x)
        seek = self.seek_time(x)
        transfer = self.transfer_time(x)
        step2 = self.step2_time(x)
        total = step1 + seek + transfer + step2
        f, ax = plt.subplots(1)
        ax.plot(np.log(x), step1, label='step 1')
        ax.plot(np.log(x), seek, label='seek')
        ax.plot(np.log(x), transfer, label='transfer')
        ax.plot(np.log(x), step2, label='step 2')
        ax.plot(np.log(x), total, label='total')
        ax.legend(prop={'size': 15})
        ax.set_title('Execution times', fontsize=12)
        ax.set_xlabel('$R$', fontsize=12)
        ax.set_ylabel('$seconds$', fontsize=12)
        f.set_size_inches(18.5, 10.5)
        plt.subplots_adjust(top=0.85)
        plt.show()

if __name__ == '__main__':
    cpu_speed = 2000 # MB/s
    seek_time_per_cluster = 0.000015 #seconds
    transfer_rate = 250 # MB/s
    light_curve_size = 5
    gpu_speed = 1
    paralelization_factor = 1
    operations_per_comparison = 3 * light_curve_size
    data_type_size = 8*10**-6
    n_clusters_function = lambda x: 25000.0/x
    n_cluster_after_pass_function = lambda x: np.exp(-0.91*np.log(x)**2+0.32*np.log(x)+6.48)
    n_lc_per_cluster_after_pass_function = lambda x: 87.86829*x**2 - 198.72312*x + 92.17059
    times = ExecutionTimes(cpu_speed, seek_time_per_cluster, transfer_rate, light_curve_size,
                           gpu_speed, n_clusters_function, n_cluster_after_pass_function, n_lc_per_cluster_after_pass_function,
                           paralelization_factor, operations_per_comparison, data_type_size)
    times.plot()