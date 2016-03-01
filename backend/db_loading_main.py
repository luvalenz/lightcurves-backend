__author__ = 'lucas'

import time
from backend.data_model.time_series import MachoFileDataBase, MultibandTimeSeries, TimeSeriesBand, TimeSeriesMongoDataBase


def load_macho_field_to_mongo(field):
    features_path = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/MACHO_features_Harmonics'
    light_curves_path = '/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/MACHO_LMC'
    macho_db = MachoFileDataBase(light_curves_path, features_path)
    mongo_db = TimeSeriesMongoDataBase('lightcurves')

    mongo_db.setup(['macho'])
    catalog_name = 'macho'
    tiles = macho_db.get_tiles_in_field(field)
    for tile in tiles:
        light_curves = macho_db.get_many(field, tile)
        if len(light_curves) != 0:
            mongo_db.add_many(catalog_name, light_curves)
        print('tile {0} added'.format(tile))

if __name__ == '__main__':
    start = time.time()
    field = 1
    load_macho_field_to_mongo(field)
    end = time.time()
    print('elapsed time = {0}'.format(end - start))


