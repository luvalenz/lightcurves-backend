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
import pymongo
from pymongo import MongoClient


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
    def add_one(self, data):
        pass

    @abstractmethod
    def add_many(self, data):
        pass



class MachoFileDataBase(TimeSeriesDataBase):

    def __init__(self, light_curves_path, features_path):
        self.light_curves_path = light_curves_path
        self.features_path = features_path

    def get_one_dict(self, id_, original=True, phase=False, features=True, metadata=True):
        field, tile, seq = id_.split('.')
        tar_path = os.path.join(self.light_curves_path, 'F_{0}'.format(field), '{0}.tar'.format(tile))
        tar = tarfile.open(tar_path)
        band_names = ['B', 'R']
        bands_dict = {}
        metadata_dict = None
        for band_name in band_names:
            band_path = 'F_{0}/{1}/lc_{2}.{3}.mjd'.format(field, tile, id_, band_name)
            try:
                light_curve_file_string = tar.extractfile(tar.getmember(band_path)).read()
                if metadata and metadata_dict is None :
                    metadata_dict = MachoFileDataBase._get_metadata_dict(field, tile, seq, light_curve_file_string)
                if original:
                    band_data_frame = pd.read_csv(StringIO(light_curve_file_string), header=2, delimiter=' ')
                    this_band_dict = MachoFileDataBase._get_band_dict(band_data_frame)
                    bands_dict[band_name] = this_band_dict
            except KeyError:
                pass
        features_dict = {}
        if features:
            features_dict = self.get_features(id_)
        return {'bands': bands_dict, 'features': features_dict, 'metadata': metadata_dict, 'id': id_}

    def get_one(self, id_, original=True, phase=False, features=True, metadata=True):
        return DataMultibandTimeSeries.from_dict(self.get_one_dict(id_, original, phase, features, metadata), self)

    @staticmethod
    def _get_metadata_dict(field, tile, seq, file_string):
        metadata_string = file_string.split('\n')[1]
        ra, dec = [float(s) for s in metadata_string.split()[3:5]]
        return {'field': field, 'tile': tile, 'seq': seq, 'ra': ra, 'dec': dec, 'catalog': 'macho'}

    @staticmethod
    def _get_band_dict(data_frame):
        times, values, errors = data_frame.values.T
        return {'times': list(times), 'values': list(values), 'errors': list(errors)}

    def get_many(self, id_list, original=True, phase=True, features=True, metadata=True):
        light_curves = []
        for id_ in id_list:
            light_curves.append(self.get_one(id_list), original, phase, features, metadata)
        return light_curves

    def get_features(self, id_):
        try:
            field, tile, seq = id_.split('.')
            tile_features_path = os.path.join(self.features_path, "F_{0}_{1}.csv".format(field, tile))
            tile_features_df = pd.read_csv(tile_features_path, sep=',', index_col=0)
            lc_features_df = tile_features_df.loc[id_]
            return lc_features_df.to_dict()
        except:
            return {}

    #can receive TimeSeries object or dict
    def update(self, id_, updated_values):
        pass

    #can receive TimeSeries object or dict
    def add_one(self, data):
        pass

    #can receive TimeSeries object or dict
    def add_many(self, data):
        pass

class TimeSeriesMongoDataBase(TimeSeriesDataBase):

    def __init__(self, url, port, db_name):
        client = MongoClient(url, port)
        self.db = client[db_name]

    def setup(self):
        collection_names = self.db.collection_names()
        for collection_name in collection_names:
            collection = self.db[collection_name]
            collection.create_index([("id", pymongo.DESCENDING)], background=True)

    def get_collection(self, catalog_name):
        return self.db[catalog_name]

    def get_one_dict(self, catalog_name, id_):
        collection = self.db[catalog_name]
        return collection.find_one({'id': id_})

    def get_one(self, catalog, id_):
        return DataMultibandTimeSeries.from_dict(self.get_one_dict(catalog, id_), self)

    def get_many(self, collection, id_list):
        return collection.find_one({'id': {'$in': id_list}})

    def metadata_search(self, catalog_name, **kwargs):
        query = []
        for key, value in kwargs.iteritems():
            query["metadata.{0}".format(key)] = value
        collection = self.db[catalog_name]
        return collection.find(query)

    def get_features(self, id_):
        return self.get_one_dict(id_)['features']

    #can receive TimeSeries object or dict
    def update(self, catalog_name, id_, updated_datum):
        collection = self.db[catalog_name]
        if not isinstance(updated_datum, dict):
            updated_datum = updated_datum.to_dict()
        collection.update_one({'id':id_}, updated_datum)

    #can receive TimeSeries object or dict
    def add_one(self, catalog_name, datum):
        collection = self.db[catalog_name]
        if not isinstance(datum, dict):
            datum = datum.to_dict()
        collection.insert_one(datum)

    def add_many(self, catalog_name, data):
        collection = self.db[catalog_name]
        data = [datum if isinstance(datum, dict) else datum.to_dict() for datum in data]
        collection.insert_many(data)

class MultibandTimeSeries(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def id(self):
        pass

    @abstractproperty
    def bands(self):
        pass

    @abstractproperty
    def band_names(self):
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
    def feature_dict(self):
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
    def phase(self):
        pass

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_band(self, band):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

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
    def bands(self):
        result = []
        for band_name in self._band_names:
            result.append(self._bands_dict[band_name])
        return result

    @property
    def band_names(self):
        return self._band_names

    @property
    def period(self):
        if 'PeriodLS' in self.feature_dict:
            return self.feature_dict['PeriodLS']
        if self.is_stored :
            self.load_features_from_db()
            if 'PeriodLS' in self.feature_dict:
                return self.feature_dict['PeriodLS']
        return self.calculate_period()

    @property
    def feature_names(self):
        return self.feature_dict.keys()

    @property
    def feature_vector(self):
        return np.array(self.feature_dict.values())

    @property
    def feature_dict(self):
        return dict(self._feature_dictionary)

    @property
    def times(self):
        result = []
        for band in self.bands:
            result.append(band.times)
        return result

    @property
    def values(self):
        result = []
        for band in self.bands:
            result.append(band.values)
        return result

    @property
    def errors(self):
        result = []
        for band in self.bands:
            result.append(band.errors)
        return result

    @property
    def phase(self):
        result = []
        if not self._is_folded:
            self.fold()
        for band in self.bands:
            result.append(band.phase)
        return result

    @property
    def metadata(self):
        return self._metadata

    @property
    def is_folded(self):
        return self._is_folded

    def __init__(self, band_names, times, values, errors=None, id_=None, phase=None, feature_dict={}, **kwargs):
        n_bands_match = len(band_names) == len(times) == len(values)
        if errors is not None:
            n_bands_match = n_bands_match and len(band_names) == len(errors)
        if phase is not None:
            n_bands_match = n_bands_match and len(band_names) == len(phase)
        if not n_bands_match:
            raise ValueError('Number of bands must match in all inputs')
        for i in range(len(band_names)):
            inputs_match = True
            if errors is not None:
                inputs_match = inputs_match and len(values[i]) == len(errors[i])
            if phase is not None:
                inputs_match = inputs_match and len(values[i]) == len(phase[i])
            if not inputs_match:
                raise ValueError('Number of entries in time and values must match. Errors and Phase must also match if any.')
        self._id = id_
        self._feature_dictionary = feature_dict
        self._id = id_
        self._bands_dict = {}
        self._is_folded = phase is not None
        self._band_names = sorted(band_names)
        for i, band_name in zip(range(len(band_names)), band_names):
            band_times = times[i]
            band_values = values[i]
            if errors is None:
                band_errors = None
            else:
                band_errors = errors[i]
            if phase is None:
                band_phase = None
            else:
                band_phase = phase[i]
            self._bands_dict[band_name] = TimeSeriesBand(self, band_name, band_times, band_values, band_errors, band_phase)

        self._metadata = kwargs.copy()

    @staticmethod
    def from_dict(dictionary):
        bands_dict = dictionary['bands']
        id_ = dictionary['id']
        times = []
        values = []
        errors = []
        phase = []
        band_names = bands_dict.keys()
        for band_name in band_names:
            this_band_dict = bands_dict[band_name]
            times.append(this_band_dict['times'])
            values.append(this_band_dict['values'])
            if errors is not None and 'errors' in this_band_dict and this_band_dict['errors'] is not None:
                errors.append(this_band_dict['errors'])
            else:
                errors = None
            if phase is not None and 'phase' in this_band_dict and this_band_dict['phase'] is not None:
                phase.append(this_band_dict['phase'])
            else:
                phase = None
        features = dictionary['features']
        metadata = dictionary['metadata']
        time_series = DataMultibandTimeSeries(band_names, times, values, errors, id_, phase, features, **metadata)
        return time_series

    def get_band(self, index_or_name):
        if type(index_or_name) is int:
            band_name = self._band_names[index_or_name]
        else:
            band_name = index_or_name
        return self._bands_dict[band_name]

    def __getitem__(self, item):
        return self.get_band(item)

    @property
    def n_bands(self):
        return len(self._band_names)

    def __len__(self):
        return self.n_bands

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
            lc = [mag, time, error, mag2, aligned_mag, aligned_mag2, aligned_time, aligned_error, aligned_error2]
            self._feature_dictionary = feature_space.calculateFeature(lc).result(method='dict')

    def calculate_period(self):
            band = self.get_band(0)
            [mag, time, error] = [band.values, band.times, band.errors]
            preprocessed_data = FATS.Preprocess_LC(mag, time, error).Preprocess()
            feature_space = FATS.FeatureSpace(Data=['magnitude','time','error'], featureList=['PeriodLS'])
            features = feature_space.calculateFeature(preprocessed_data).result(method='dict')
            self._feature_dictionary['PeriodLS'] = features['PeriodLS']

    def load_features_from_db(self, database):
            self._feature_dictionary = database.get_features(self.id)

    def to_dict(self):
        dictionary = []
        bands = {}
        for band_name in self._band_names:
            band = self.get_band(band_name)
            bands[band_name] = band.to_dict()
        dictionary['features'] = self._feature_dictionary.copy()
        dictionary['bands'] = bands
        dictionary['metadata'] = self._metadata.copy()
        dictionary['id'] = self.id
        return dictionary

    def fold(self):
        for band in self.bands:
            band.fold()
        self._is_folded = True


class SyntheticTimeSeries(MultibandTimeSeries):
    pass


class TimeSeriesBand(object):

    @property
    def time_series(self):
        return self._time_series

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
    def times_values(self):
        return np.column_stack((self.times, self.values))

    @property
    def times_values_errors(self):
        if self._errors is None:
            return self.times_values
        return np.column_stack((self.times, self.values, self.errors))

    @property
    def has_errors(self):
        return self._errors is not None

    @property
    def phase(self):
        return self._phase

    @property
    def is_folded(self):
        return self._phase is not None

    @property
    def period(self):
        return self.time_series.period

    def __init__(self, time_series, name, times, values, errors=None, phase=None):
        self._time_series = time_series
        self._name = name
        self._times = np.array(times)
        self._values = np.array(values)
        if errors is not None:
            errors = np.array(errors)
        if phase is not None:
            phase = np.array(phase)
        self._errors = errors
        self._phase = phase

    def fold(self):
        t = self.period
        self._phase = np.mod(self.times, t) / t

    def to_dict(self):
        return {'times': self.times, 'values': self.values, 'errors': self.errors, 'phase': self._phase}

    def to_array(self):
        return np.column_stack((self.times, self.values, self.errors))






