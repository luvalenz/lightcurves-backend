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
import glob
import pickle
from bson.binary import Binary
import itertools


class TimeSeriesDataBase(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def catalog_names(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def defragment(self):
        pass

    @abstractmethod
    def get_one(self, id_, original=True, phase=True, features=True, metadata=True):
        pass

    @abstractmethod
    def get_many(self, id_list, original=True, phase=True, features=True, metadata=True):
        pass

    @abstractmethod
    def get_all(self):
        return

    @abstractmethod
    def update_one(self, time_series):
        pass

    @abstractmethod
    def update_many(self, time_series_list):
        pass

    @abstractmethod
    def add_one(self, data):
        pass

    @abstractmethod
    def add_many(self, catalog, data):
        pass

    @abstractmethod
    def delete_one(self, catalog, id_):
        pass

    @abstractmethod
    def delete_catalog(self, catalog):
        pass

    @abstractmethod
    def get_original_bands(self, id_):
        pass

    @abstractmethod
    def get_features(self, id_):
        pass

    @abstractmethod
    def get_metadata(self, id_):
        pass

    @abstractmethod
    def get_reduced_vector(self, id_):
        pass


class MachoFileDataBase(TimeSeriesDataBase):

    def __init__(self, light_curves_path, features_path):
        self.light_curves_path = light_curves_path
        self.features_path = features_path

    @property
    def catalog_names(self):
        pass

    def setup(self):
        pass

    def defragment(self):
        pass

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
        return {'bands': bands_dict, 'features': features_dict, 'metadata': metadata_dict, 'id': 'macho.{0}'.format(id_)}

    def get_one(self, id_, original=True, phase=False, features=True, metadata=True):
        return DataMultibandTimeSeries.from_dict(self.get_one_dict(id_, original, phase, features, metadata))

    def get_many_dict(self, field, tile, original=True, phase=False, features=True, metadata=True):
        tar_path = os.path.join(self.light_curves_path, 'F_{0}'.format(field), '{0}.tar'.format(tile))
        tar = tarfile.open(tar_path)
        paths = tar.getnames()
        ids = [path.split('_')[-1][:-6] for path in paths if path.endswith('.mjd')]
        ids = sorted(list(set(ids)))
        band_names = ['B', 'R']
        bands_dict = {}
        metadata_dict = None
        list_of_dicts = []
        for id_ in ids:
            seq = id_.split('.')[-1]
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
            list_of_dicts.append({'bands': bands_dict, 'features': features_dict, 'metadata': metadata_dict, 'id': id_})
        return list_of_dicts

    def get_all(self, n_fields=82):
        return MachoTimeSeriesIterator(True, self, n_fields)

    def get_many(self, field, tile, original=True, phase=False, features=True, metadata=True):
        list_of_dicts = self.get_many_dict(field, tile, original, phase, features, metadata)
        list_of_time_series = []
        for dictionary in list_of_dicts:
            list_of_time_series.append(DataMultibandTimeSeries.from_dict(dictionary))
        return list_of_time_series

    @staticmethod
    def _get_metadata_dict(field, tile, seq, file_string):
        metadata_string = file_string.split('\n')[1]
        ra, dec = [float(s) for s in metadata_string.split()[3:5]]
        return {'field': field, 'tile': tile, 'seq': seq, 'ra': ra, 'dec': dec, 'catalog': 'macho'}

    @staticmethod
    def _get_band_dict(data_frame):
        times, values, errors = data_frame.values.T
        return {'times': list(times), 'values': list(values), 'errors': list(errors)}

    #can receive TimeSeries object or dict
    def update_one(self, updated_datum, catalog_name, id_):
        pass

    def update_many(self, time_series_list):
        pass

    #can receive TimeSeries object or dict
    def add_one(self, data):
        pass

    #can receive TimeSeries object or dict
    def add_many(self, data):
        pass

    def delete_one(self, catalog, id_):
        pass

    def delete_catalog(self, catalog):
        pass

    def get_tiles_in_field(self, field):
        field_path = os.path.join(self.light_curves_path, 'F_{0}'.format(field))
        tars = glob.glob(os.path.join(field_path, '*.tar'))
        return sorted([int(os.path.basename(file_name).split('.')[0]) for file_name in tars])

    def get_features(self, id_):
        try:
            field, tile, seq = id_.split('.')
            tile_features_path = os.path.join(self.features_path, "F_{0}_{1}.csv".format(field, tile))
            tile_features_df = pd.read_csv(tile_features_path, sep=',', index_col=0)
            lc_features_df = tile_features_df.loc[id_]
            return lc_features_df.to_dict()
        except:
            return {}

    def get_metadata(self, id_):
        field, tile, seq = id_.split('.')
        return {'field': field, 'tile':tile, 'seq': seq, 'catalog': 'macho'}

    def get_reduced_vector(self, id_):
        return None

    def get_original_bands(self, id_):
        pass


class MongoTimeSeriesDataBase(TimeSeriesDataBase):

    def __init__(self, batch_size=3*10**5, db_name='lightcurves', url='localhost', port=27017):
        client = MongoClient(url, port)
        self.db = client[db_name]
        self._batch_size = batch_size
        self._current_catalog = None
        self._current_cursor = None

    @property
    def catalog_names(self):
        collection_names = self.db.collection_names()
        if 'system.indexes' in collection_names:
            collection_names.remove('system.indexes')
        return collection_names

    #must be called after adding elements to the db
    def setup(self):
        collection_names = self.catalog_names
        for collection_name in collection_names:
            collection = self.db[collection_name]
            collection.create_index([("id", pymongo.ASCENDING)], background=True, unique=True)

    def defragment(self):
        collection_names = self.db.collection_names()
        collection_names.remove('system.indexes')
        results = {}
        for collection_name in collection_names:
            results[collection_name] = self.db.command('compact', collection_name)
        return results

    def get_one_dict(self, catalog_name, id_):
        collection = self.db[catalog_name]
        return collection.find_one({'id': id_})

    def get_one(self, catalog, id_):
        dictionary = self.get_one_dict(catalog, id_)
        if dictionary is not None:
            return DataMultibandTimeSeries.from_dict(dictionary)
        else:
            return None

    def get_all(self, batch_size=None):
        return self.find_many(None, {}, True, batch_size)

    def find_many(self, catalog_name, query_dict, batch=True, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size
        if catalog_name is None:
            catalogs = self.catalog_names
        else:
            catalogs = [catalog_name]
        cursors = []
        for catalog in catalogs:
            collection = self.db[catalog]
            cursor = collection.find(query_dict)
            cursors.append(cursor)
        return MongoTimeSeriesIterator(cursors, batch, batch_size)

    def get_many(self, id_list, catalog_name=None, batch=True, batch_size=None):
        query = {'id': {'$in': id_list}}
        return self.find_many(catalog_name, query, batch, batch_size)

    def metadata_search(self, catalog_name, **kwargs):
        query = []
        for key, value in kwargs.iteritems():
            query["metadata.{0}".format(key)] = value
        collection = self.db[catalog_name]
        return collection.find(query)

    #receives TimeSeries
    def update_one(self, time_series):
        collection = self.db[time_series.catalog]
        updated_datum = time_series.to_dict()
        collection.replace_one({'id':time_series.id}, updated_datum)

    def update_many(self, time_series_list):
        for time_series in time_series_list:
            self.update_one(time_series)

    #can receive TimeSeries object or dict
    def add_one(self, catalog_name, datum):
        collection = self.db[catalog_name]
        if not isinstance(datum, dict):
            datum = datum.to_dict()
        collection.insert_one(datum)

    def add_many(self, catalog_name, data):
        collection = self.db[catalog_name]
        ids = []
        data_dicts = []
        for datum in data:
            datum_dict = datum if isinstance(datum, dict) else datum.to_dict()
            data_dicts.append(datum_dict)
            ids.append(datum_dict['id'])
        collection.insert_many(data_dicts)
        return ids

    def delete_one(self, catalog_name, id_):
        collection = self.db[catalog_name]
        collection.delete_one({'id': id_})

    def delete_catalog(self, catalog_name):
        self.db[catalog_name].drop()

    def get_features(self, catalog_name, id_):
        return self.get_one_dict(catalog_name, id_)['features']

    def get_metadata(self, catalog_name, id_):
        return self.get_one_dict(catalog_name, id_)['metadata']

    def get_reduced_vector(self, catalog_name, id_):
        return self.get_one_dict(catalog_name, id_)['reduced']

    def get_original_bands(self, catalog_name, id_):
        bands_dict = self.get_one_dict(catalog_name, id_)['bands']
        return DataMultibandTimeSeries.extract_bands_dict(bands_dict)


class MultibandTimeSeries(object):
    __metaclass__ = ABCMeta

    @abstractproperty
    def id(self):
        pass

    @abstractmethod
    def catalog(self):
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
    def catalog(self):
        return self._metadata['catalog']

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
        else:
            self.calculate_period()
            return self.feature_dict['PeriodLS']

    @property
    def feature_names(self):
        sorted_features = sorted(self._feature_dictionary.items(), key=lambda x:x[0])
        return [key for (key, value) in sorted_features]

    @property
    def feature_vector(self):
        sorted_features = sorted(self._feature_dictionary.items(), key=lambda x:x[0])
        return np.array([value for (key, value) in sorted_features])

    @property
    def feature_dict(self):
        return dict(self._feature_dictionary)

    @property
    def reduced_vector(self):
        rv = self._reduced_vector
        return np.array(rv) if rv is not None else None

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

    def __init__(self, band_names=None, times=None, values=None,
                 errors=None, id_=None, phase=None, reduced_vector=None,
                 feature_dict=None, metadata_dict=None):
        if band_names is None:
            band_names = []
        if feature_dict is None:
            feature_dict = {}
        if metadata_dict is None:
            feature_dict = {}
        self._set_bands(band_names, times, values, errors, phase)
        self._set_features(feature_dict)
        self._set_metadata(metadata_dict)
        self.set_reduced(reduced_vector)
        self._id = id_

    def _set_bands(self, band_names, times, values, errors, phase):
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
                raise ValueError('Number of entries in time and values must match. '
                                 'Errors and Phase must also match if any.')
        self._band_names = band_names
        self._bands_dict = {}
        self._is_folded = phase is not None
        if band_names is not None:
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
                self._bands_dict[band_name] = TimeSeriesBand(self, band_name, band_times,
                                                             band_values, band_errors, band_phase)

    def _set_features(self, feature_dict):
        self._feature_dictionary = feature_dict.copy() if feature_dict is not None else None

    def _set_metadata(self, metadata_dict):
        self._metadata = metadata_dict.copy() if metadata_dict is not None else None

    def set_reduced(self, reduced_vector):
        self._reduced_vector = list(reduced_vector) if reduced_vector is not None else None

    @staticmethod
    def extract_bands_dict(bands_dict):
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
        return band_names, times, values, errors, phase

    @staticmethod
    def from_dict(dictionary):
        if 'bands' in dictionary:
            band_names, times, values, errors, phase = DataMultibandTimeSeries.extract_bands_dict(dictionary['bands'])
        else:
            times = values = errors = phase = band_names = None
        id_ = dictionary['id']  if 'id' in dictionary else None
        features = dictionary['features'] if 'features' in dictionary else None
        metadata = dictionary['metadata'] if 'metadata' in dictionary else None
        reduced_vector = dictionary['reduced'] if 'reduced' in dictionary else None
        time_series = DataMultibandTimeSeries(band_names, times, values, errors, id_, phase, reduced_vector, features, metadata)
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

    def load_metadata_from_db(self, database):
        self._metadata_dictionary = database.get_metadata(self.catalog, self.id)

    def load_features_from_db(self, database):
            self._feature_dictionary = database.get_features(self.catalog, self.id)

    def load_bands_from_db(self, database):
        band_names, times, values, errors, phase = database.get_original_bands(self.catalog, self.id)
        self._set_bands(band_names, times, values, errors, phase)

    def load_reduced_vector_from_db(self, database):
        self.set_reduced(database.get_reduced_vector(self.catalog, self.id))

    def to_dict(self):
        dictionary = {}
        bands = {}
        for band_name in self._band_names:
            band = self.get_band(band_name)
            bands[band_name] = band.to_dict()
        dictionary['features'] = self._feature_dictionary.copy()
        dictionary['bands'] = bands
        dictionary['metadata'] = self._metadata.copy()
        dictionary['id'] = self.id
        dictionary['reduced'] = list(self._reduced_vector) if self._reduced_vector is not None else None
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
        times = self.times.tolist()
        values = self.values.tolist()
        errors = self.errors.tolist() if self.errors is not None else None
        phase = self.phase.tolist() if self.phase is not None else None
        return {'times': times, 'values': values, 'errors': errors, 'phase': phase}

    def to_array(self):
        return np.column_stack((self.times, self.values, self.errors))


class TimeSeriesIterator(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, batch= True, batch_size=3*10**5):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def rewind(self):
        pass


class MongoTimeSeriesIterator(TimeSeriesIterator):

    def __init__(self, cursors, batch=True, batch_size=3*10**5):
        self._batch = batch
        self._batch_size = batch_size
        self._cursors = [cursor.clone() for cursor in cursors]
        self._current_cursor_index = 0

    def __len__(self):
        length = 0
        for cursor in self._cursors:
            length += cursor.count()
        return length

    def __iter__(self):
        return self

    def next_unit(self):
        while True:
            if self._current_cursor_index >= len(self._cursors):
                raise StopIteration
            try:
                time_series_dict = self._current_cursor.next()
                return DataMultibandTimeSeries.from_dict(time_series_dict)
            except StopIteration:
                self._current_cursor_index += 1

    def next_batch(self):
        time_series_batch = []
        i = 0
        while True:
            try:
                time_series = self.next_unit()
                time_series_batch.append(time_series)
            except StopIteration:
                break
            if i == self._batch_size - 1:
                break
            i += 1
        if len(time_series_batch) == 0:
            raise StopIteration
        return time_series_batch

    def next(self):
        if self._batch:
            return self.next_batch()
        else:
            return self.next_unit()

    def rewind(self):
        for cursor in self._cursors:
            cursor.rewind()
        self._current_cursor_index = 0

    @property
    def _current_cursor(self):
        return self._cursors[self._current_cursor_index]


class MachoTimeSeriesIterator(TimeSeriesIterator):

    def __init__(self, batch, database, n_fields=82):
        self._batch = batch
        self._database = database
        self._n_fields = n_fields
        self.rewind()


    def __len__(self):
        pass

    def __iter__(self):
        return self

    def next_unit(self):
        return None

    def next_batch(self):
        if self._current_tile_index >= len(self._current_field_tiles):
            self._current_tile_index = 0
            self._current_field_tiles = self._database.get_tiles_in_field(self._current_field)
            self._current_field += 1
        if self._current_field > self._n_fields:
            raise StopIteration
        batch = self._database.get_many(self._current_field, self._current_field_tiles[self._current_tile_index])
        self._current_tile_index += 1
        return batch

    def next(self):
        if self._batch:
            return self.next_batch()
        else:
            return self.next_unit()

    def rewind(self):
        self._current_field = 1
        self._current_tile_index = 0
        self._current_field_tiles = self._database.get_tiles_in_field(self._current_field)




