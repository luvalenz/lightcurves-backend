__author__ = 'lucas'

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import OrderedDict
import numpy as np


class TimeSeriesDataBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_one(self, id_):
        pass

    @abstractmethod
    def get_many(self, id_list):
        pass

    #can receive TimeSeries object or dict
    @abstractmethod
    def update(self, id_, updated_values):
        pass

    #can receive TimeSeries object or dict
    @abstractmethod
    def add(self, data):
        pass


class TimeSeriesFileDataBase(TimeSeriesDataBase):
    pass


class TimeSeriesMongoDataBase(TimeSeriesDataBase):
    pass


class MultibandTimeSeries(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def period(self):
        pass

    @abstractproperty
    def feature_names(self):
        pass

    @abstractproperty
    def feature_vector(self):
        pass

    @abstractproperty
    def feature_dictionary(self):
        pass

    @abstractproperty
    def is_interpolated(self):
        pass

    @abstractproperty
    def is_folded(self):
        pass

    @abstractproperty
    def interpolation_resolution(self):
        pass

    @abstractproperty
    def times(self):
        pass

    @abstractproperty
    def values(self):
        pass

    @abstractproperty
    def errors(self):
        pass

    @abstractproperty
    def interpolated_time(self):
        pass

    @abstractproperty
    def interpolated_values(self):
        pass

    @abstractproperty
    def folded_phase(self):
        pass

    @abstractproperty
    def folded_values(self):
        pass

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_band(self, band):
        pass

    @abstractmethod
    def __getitem__(self, item):
        return self.get_band(item)

    @abstractproperty
    def n_bands(self):
        pass

    @abstractmethod
    def __len__(self):
        return self.n_bands

    @abstractmethod
    def load_features(self):
        pass

    @abstractmethod
    def interpolate(self):
        pass

    @abstractmethod
    def fold(self):
        pass

    @abstractmethod
    def is_stored(self):
        pass

    @abstractmethod
    def update(self):
        pass


class DataMultibandTimeSeries(MultibandTimeSeries):
    @property
    def period(self):
        if 'PeriodLS' in self.feature_dictionary:
            return self.feature_dictionary['PeriodLS']
        ##TODO elif hay base de datos y está ahí entoces sacar de ahí else calcular con fats

    @property
    def feature_names(self):
        return self.feature_dictionary.keys()

    @property
    def feature_vector(self):
        return np.array(self.feature_dictionary.values())

    @property
    def feature_dictionary(self):
        return self._feature_dictionary

    @property
    def is_interpolated(self):
        pass

    @property
    def is_folded(self):
        pass

    @property
    def interpolation_resolution(self):
        pass

    @property
    def times(self):
        return self.times

    @property
    def values(self):
        result = []
        for band in self.bands:
            result.append(band.values)
        return np.column_stack(result)


    @property
    def errors(self):
        result = []
        for band in self.bands:
            result.append(band.errors)
        return np.column_stack(result)


    @property
    def interpolated_times(self):
        pass

    @property
    def interpolated_values(self):
        pass

    @property
    def folded_phase(self):
        pass

    @property
    def folded_values(self):
        pass

    def __init__(self, band_names, times, values, errors=None, id_=None, database=None):
        if values.ndim == 1:
            values = np.array(np.matrix(values).T)
        if errors is not None and errors.ndim == 1:
            errors = np.array(np.matrix(values).T)
        times_length = len(times)
        n_bands = len(band_names)
        values_dim_i, values_dim_j = values.shape
        if errors is None:
            errors_dim_i, errors_dim_j = values_dim_i, values_dim_j
        else:
            errors_dim_i, errors_dim_j = errors.shape
        if not (times_length == values_dim_i == errors_dim_i and values_dim_j == errors_dim_j == n_bands):
            raise ValueError('Dimensions in inputs must match')
        self.bands = OrderedDict()
        self.times = times
        for band_name, i in zip(band_names, range(len(band_names))):
            band_values = values[:,i]
            if errors is None:
                band_errors = None
            else:
                band_errors = errors[:,i]
            self.bands[band_names] = TimeSeriesBand(band_values, band_errors)
        self.is_interpolated = False
        self.is_folded = False
        self.id = id_
        self.database = database

    @staticmethod
    def from_dictionary(dictionary):
        times = np.array(dictionary['times'])
        bands = dictionary['bands']
        band_names = []
        values = []
        errors = []
        for band in bands:
            band_names.append(band['band_names'])
            values.append(band['values'])
            if errors is not None and 'errors' in band and band['errors'] is not None:
                errors.append(band['errors'])
            else:
                errors = None
        values = np.column_stack(values)
        if errors is not None:
            errors = np.column_stack(errors)
        return DataMultibandTimeSeries(band_names, times, values, errors)

    def get_band(self, index_or_name):
        if type(index_or_name) is int:
            return self.bands.values()[index_or_name]
        else:
            return self.bands[index_or_name]


    def __getitem__(self, item):
        return super(MultibandTimeSeries, self).__getitem__(self, item)

    @property
    def n_bands(self):
        return len(self.bands)

    def __len__(self):
        return self.n_bands

    def load_features(self):
        pass

    def interpolate(self):
        pass

    def fold(self):
        pass

    def is_stored(self):
        pass

    def update(self):
        pass


class SyntheticTimeSeries(MultibandTimeSeries):
    pass


class TimeSeriesBand(object):


    @property
    def values(self):
        pass

    @property
    def errors(self):
        pass

    @property
    def has_errors(self):
        pass

    @property
    def resolution(self):
        pass

    @property
    def interpolated_values(self):
        pass

    @abstractproperty
    def interpolation_resolution(self):
        pass


    @abstractmethod
    def __init__(self, values):
        pass

    @abstractmethod
    def interpolate(self):
        pass

    @abstractmethod
    def fold(self):
        pass


