__author__ = 'lucas'

from backend.data_model.time_series import DataMultibandTimeSeries
from backend.data_model.data_model_interface import DataModelInterface, load_config
from backend.online.online_algorithms import OurMethod
import matplotlib.pyplot as plt
import numpy as np


class OnlineInterface(object):

    def __init__(self, data_model_interface):
        clustering_db = data_model_interface.get_clustering_database()
        time_series_db = data_model_interface.get_time_series_database()
        self._indexing_model = OurMethod(clustering_db, time_series_db)
        self._reduction_model = data_model_interface.get_reduction_model()

    def feature_space_query(self, feature_dict, n_neighbours):
        time_series_target = DataMultibandTimeSeries.from_dict({'features': feature_dict})
        self._reduction_model.transform_time_series([time_series_target])
        #print time_series_target.reduced_vector
        return self._indexing_model.time_series_query(time_series_target, n_neighbours)

    def time_series_space_query(self, time_series):
        pass


def plot_time_series(time_series, color='blue'):
    time_series.fold()
    plt.plot(time_series[1].phase, time_series[1].values, '*', color=color)
    plt.show()

if __name__ == '__main__':
    config = load_config('/home/lucas/PycharmProjects/lightcurves-backend/backend/config.json')
    data_model_interface = DataModelInterface(config)
    online_interface = OnlineInterface(data_model_interface)
    ts_target = data_model_interface.get_time_series_database().get_one('macho', '1.3447.36')
    print ts_target.reduced_vector
    feature_dict = ts_target.feature_dict
    plot_time_series(ts_target, 'red')
    time_series_ranking, distances = online_interface.feature_space_query(feature_dict, 10)
    for ts, distance in zip(time_series_ranking, distances):
        plot_time_series(ts)


