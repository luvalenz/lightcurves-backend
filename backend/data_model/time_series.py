__author__ = 'lucas'

from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import FATS
import os
import pandas as pd
import tarfile
import sys
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO


class TimeSeriesDataBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_one(self, id_, original=True, phase=True, features=True, metadata=True):
        pass

    @abstractmethod
    def get_many(self, id_list, original=True, phase=True, features=True, metadata=True):
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


class MachoFileDataBase(TimeSeriesDataBase):

    def __init__(self, light_curves_path, features_path):
        self.light_curves_path = light_curves_path
        self.features_path = features_path

    def get_one(self, id_, original=True, phase=True, features=True, metadata=True):
        field, tile, seq = id_.split('.')
        tar_path = os.path.join(self.light_curves_path, 'F_{0}'.format(field), '{0}.tar'.format(tile))
        tar = tarfile.open(tar_path)
        band_names = ['B', 'R']
        bands = {}
        for band in bands:
            band_path = 'F_{0}/{1}/lc_{2}.{3}.mjd'.format(field, tile, id_, band)
            try:
                light_curve_file_string = tar.extractfile(tar.getmember(band_path)).read()

            except KeyError:
                pass

            return pd.read_csv(StringIO(light_curve_file_string), header=2, delimiter=' ')

    def get_many(self, id_list, original=True, phase=True, features=True, metadata=True):
        light_curves = []
        for id_ in id_list:
            light_curves.append(self.get_one(id_list), original, phase, features, metadata)
        return light_curves

    def get_features(self, id_):
        pass

    #can receive TimeSeries object or dict
    def update(self, id_, updated_values):
        pass

    #can receive TimeSeries object or dict
    def add(self, data):
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

    @property
    def times(self):
        return self._times

    @property
    def values(self):
        return self._values

    @property
    def errors(self):
        return self._errors

    @property
    def phase(self):
        if self._phase is None:
            self.fold()
        return self._phase

    @property
    def is_stored(self):
        return self.database is not None

    @property
    def database(self):
        return self._database

    @property
    def metadata(self):
        return self._metadata

    def __init__(self, band_names, times, values, errors=None, id_=None, phase=None, feature_dict={}, **kwargs):
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
        self._id = id_
        self._feature_dictionary = feature_dict
        self._phase = phase
        self._phase = phase
        self._id = None
        self._database = None
        self._band_names = band_names
        self._bands = {}
        for i, band_name in zip(range(len(band_names)), band_names):
            band_times = times[:, i]
            band_values = values[:, i]
            if errors is None:
                band_errors = None
            else:
                band_errors = errors[:, i]
            self._bands[band_name] = TimeSeriesBand(band_times, band_values, band_errors)
        self._metadata = kwargs.copy()

    @staticmethod
    def from_dict(dictionary):
        bands = dictionary['bands']
        id_ = dictionary['id']
        band_names = []
        times = []
        values = []
        errors = []
        phase = []
        for band in bands:
            band_names.append(band['band_names'])
            times.append(band['times'])
            values.append(band['values'])
            if errors is not None and 'errors' in band and band['errors'] is not None:
                errors.append(band['errors'])
            else:
                errors = None
            if phase is not None and 'phase' in band and band['phase'] is not None:
                phase.append(band['phase'])
            else:
                phase = None
        times = np.column_stack(times)
        values = np.column_stack(values)
        if errors is not None:
            errors = np.column_stack(errors)
        if phase is not None:
            phase = np.column_stack(phase)
        features = dictionary['features']
        metadata = dictionary['metadata'].copy()
        time_series = DataMultibandTimeSeries(band_names, times, values, errors, id_, phase, features, **metadata)
        return time_series

    def get_band(self, index_or_name):
        if type(index_or_name) is int:
            band_name = self._band_names[index_or_name]
        else:
            band_name = index_or_name
        return self._bands[band_name]

    def __getitem__(self, item):
        return super(MultibandTimeSeries, self).__getitem__(self, item)

    @property
    def n_bands(self):
        return len(self._band_names)

    def __len__(self):
        return self.n_bands

    def update(self):
        if self.is_stored:
            self.database.update(self.id, self.to_dict())

    def calculate_features(self):
        if self.n_bands == 1:
            band = self.get_band(0)
            [mag, time, error] = [band.values, band.times, band.errors]
            [mag, time, error] = FATS.Preprocess_LC(mag, time, error).Preprocess()
            feature_space = FATS.FeatureSpace(Data=['magnitude','time','error'], featureList=None)
            self._feature_dictionary = feature_space.calculateFeature([mag, time, error]).result(method='dict')
        else:
            band1 = self.get_band(0)
            band2 = self.get_band(1)
            feature_space = FATS.FeatureSpace(Data='all', featureList=None)
            [mag, time, error] = [band1.values, band1.times, band1.errors]
            [mag2, time2, error2] = [band2.values, band2.times, band2.errors]
            [mag, time, error] = FATS.Preprocess_LC(mag, time, error).Preprocess()
            [mag2, time2, error2] = FATS.Preprocess_LC(mag2, time2, error2).Preprocess()
            [aligned_mag, aligned_mag2, aligned_time, aligned_error, aligned_error2] = FATS.Align_LC(time, time2, mag, mag2, error, error2)
            lc = [mag, time, error] + [aligned_mag, aligned_mag2, aligned_time, aligned_error, aligned_error2]
            self._feature_dictionary = feature_space.calculateFeature(lc).result(method='dict')

    def calculate_period(self):
            band = self.get_band(0)
            [mag, time, error] = [band.values, self.times, band.errors]
            preprocessed_data = FATS.Preprocess_LC(mag, time, error).Preprocess()
            feature_space = FATS.FeatureSpace(Data=['magnitude','time','error'], featureList=['PeriodLS'])
            features = feature_space.calculateFeature(preprocessed_data).result(method='dict')
            self._feature_dictionary['PeriodLS'] = features['PeriodLS']

    def load_features_from_db(self):
        if self.database is not None:
            self._feature_dictionary = self.database.get_features(self.id)

    def to_dict(self):
        dictionary = []
        bands = {}
        for band_name in self._band_names:
            band = self.get_band(band_name)
            bands[band_name] = band.to_dict()
        dictionary['features'] = self._feature_dictionary.copy()
        dictionary['bands'] = bands
        dictionary['metadata'] = self._metadata.copy()

    def fold(self):
        t = self.period
        self._phase = np.mod(self.times, t) / t


class SyntheticTimeSeries(MultibandTimeSeries):
    pass


class TimeSeriesBand(object):

    @property
    def name(self):
        return self._name

    @property
    def times(self):
        return self._times

    @property
    def values(self):
        return self._values

    @property
    def errors(self):
        return self._errors

    @property
    def has_errors(self):
        return self._errors is not None

    @property
    def phase(self):
        return self._phase

    def __init__(self, name, times, values, errors=None, phase=None):
        self._name = name
        self._times = times
        self._values = np.array(values)
        if errors is None:
            errors = np.array(errors)
        if phase is None:
            phase = np.array(phase)
        self._errors = errors
        self._phase = phase

    def to_dict(self):
        return {'times': self.times, 'values': self.values, 'errors': self.errors, 'phase': self._phase}

    def to_array(self):
        return np.column_stack((self.times, self.values, self.errors))






