__author__ = 'lucas'
from offline.offline_interface import get_macho_field, add_data

if __name__ == '__main__':
    macho_field = get_macho_field('/media/lucas/115d830f-0d51-49ad-8a2f-84544fbab639/MACHO_features_Harmonics', 1)
    print(len(macho_field))
    macho_field = macho_field[:500]
    add_data(macho_field)