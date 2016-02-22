__author__ = 'lucas'

from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import FATS


class TimeSeriesDataBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_one(self, id_, original=True, phase=True, features=True):
        pass

    @abstractmethod
    def get_many(self, id_list, original=True, phase=True, features=True):
        pass

    @abstractmethod
    def get_features(self, id_):
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
    def id(self):
        pass

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

    # @abstractproperty
    # def is_interpolated(self):
    #     pass

    # @abstractproperty
    # def interpolation_resolution(self):
    #     pass

    @abstractproperty
    def times(self):
        pass

    @abstractproperty
    def values(self):
        pass

    @abstractproperty
    def errors(self):
        pass

    # @abstractproperty
    # def interpolated_time(self):
    #     pass

    # @abstractproperty
    # def interpolated_values(self):
    #     pass

    @abstractproperty
    def phase(self):
        pass

    @abstractproperty
    def is_stored(self):
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
    def update(self):
        pass

    @abstractmethod
    def calculate_features(self):
        pass

    @abstractmethod
    def calculate_period(self):
        pass

    @abstractmethod
    def to_dict(self):
        pass

    @abstractmethod
    def fold(self):
        pass

    # @abstractmethod
    # def interpolate(self):
    #     pass


class DataMultibandTimeSeries(MultibandTimeSeries):

    @property
    def id(self):
        return self._id

    @property
    def period(self):
        if 'PeriodLS' in self.feature_dictionary:
            return self.feature_dictionary['PeriodLS']
        if self.is_stored :
            self.load_features_from_db()
            if 'PeriodLS' in self.feature_dictionary:
                return self.feature_dictionary['PeriodLS']
        return self.calculate_period()

    @property
    def feature_names(self):
        return self.feature_dictionary.keys()

    @property
    def feature_vector(self):
        return np.array(self.feature_dictionary.values())

    @property
    def feature_dictionary(self):
        return dict(self._feature_dictionary)

    # @property
    # def is_interpolated(self):
    #     pass

    # @property
    # def interpolation_resolution(self):
    #     pass

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

    # @property
    # def interpolated_times(self):
    #     pass
    #
    # @property
    # def interpolated_values(self):
    #     pass

    @property
    def phase(self):
        if self._phase is None:
            self.fold()
        return self._phase

    @property
    def is_stored(self):
        return self.database is not None

    def __init__(self, band_names, times, values, errors=None, phase=None, feature_dict={}):
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
        if errors is not None:
            errors = np.array(errors)
        if phase is not None:
            phase = np.array(phase)
        # self.is_interpolated = False
        self._feature_dictionary = feature_dict
        self._times = np.array(times)
        self._errors = errors
        self._phase = phase
        self._phase = phase
        self._id = None
        self.database = None
        self.band_names = band_names

    @staticmethod
    def from_dict(dictionary):
        times = dictionary['times']
        bands = dictionary['bands']
        phase = dictionary['phase']
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
        features = dictionary['features']
        time_series = DataMultibandTimeSeries(band_names, times, values, errors, phase, features)
        return time_series

    def get_band(self, index_or_name):
        if type(index_or_name) is int:
            index = index_or_name
            band_name = self.band_names[index_or_name]
        else:
            index = self.band_names.index(index_or_name)
            band_name = index_or_name
        values = self.values[:, index]
        if self.errors is None:
            errors = None
        else:
            errors = self.errors[:, index]
        return TimeSeriesBand(band_name, self, values, errors)

    def __getitem__(self, item):
        return super(MultibandTimeSeries, self).__getitem__(self, item)

    @property
    def n_bands(self):
        return len(self.band_names)

    def __len__(self):
        return self.n_bands

    def update(self):
        if self.is_stored:
            self.database.update(self.id, self.to_dict())

    def calculate_features(self):
        #TODO USE FATS

    def calculate_period(self):
        #TODO USE FATS

    def load_features_from_db(self):
        if self.database is not None:
            self.features = self.database.get_features(self.id)

    def to_dict(self):
        dictionary = []
        dictionary['times'] = list(self.times)
        dictionary['phase'] = self._phase
        values = []
        errors = []
        bands = {}
        for band_name in self.band_names:
            self
        dictionary['features'] = self._feature_dictionary
        dictionary['bands'] = band_name

    def fold(self):
        t = self.period
        self._phase = np.mod(self.times, t) / t


class SyntheticTimeSeries(MultibandTimeSeries):
    pass


class TimeSeriesBand(object):
    @property
    def times(self):
        return self.time_series.times

    @property
    def values(self):
        return self._values

    @property
    def period(self):
        return self.time_series.period

    @property
    def errors(self):
        return self._errors

    @property
    def has_errors(self):
        return self._errors is not None

    # @property
    # def interpolated_values(self):
    #     pass

    # @property
    # def interpolation_resolution(self):
    #     self.time_series.interpolation_resolution

    def __init__(self, name, time_series, values, errors=None):
        self.name = name
        self._values = np.array(values)
        if errors is None:
            errors = np.array(errors)
        self._errors = errors
        self.time_series = time_series

    def to_dict(self):
        return {'values': self.values, 'errors': self.errors}

    def to_array(self):
        return np.column_stack((self.times, self.values, self.errors))






