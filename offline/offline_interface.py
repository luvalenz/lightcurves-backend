__author__ = 'lucas'

import yaml, os, glob
import pandas as pd
import numpy as np
from offline_data_structures import Birch, IncrementalPCA as IPCA

def get_config():
    print(glob.glob('*'))
    return yaml.safe_load(open('config.yaml'))


#get the data and adds the 'macho_' to the index of every lightcurve
def get_macho_field(dataset_path, field):
    dataframes = []
    file_paths = glob.glob("{0}/F_{1}_*".format(dataset_path, field))
    for file_path in file_paths:
        file_data = pd.read_csv(file_path, sep=',', index_col=0)
        dataframes.append(file_data)
    resulting_dataframe = pd.concat(dataframes)
    resulting_dataframe.index = np.core.defchararray.add('macho_', resulting_dataframe.index.values)
    return resulting_dataframe

def add_data(dataframe):
    configuration = get_config()
    path = configuration['clusters_path']
    radius = configuration['radius']
    dimensionality_reduction = configuration['dimensionality_reduction']
    dimensionality = configuration['dimensionality']
    full_path = os.path.join(path, '{0}_{1}'.format(dimensionality_reduction, dimensionality), str(radius))
    if not os.path.exists(full_path):
        try:
            os.makedirs(full_path)
        except OSError:
            print("Directory already exists")
    birch_path = os.path.join(full_path, 'birch.pkl')
    reduction_model_path = os.path.join(full_path, 'reduction_model.pkl')
    if os.path.isfile(birch_path):
        birch = Birch.from_pickle(birch_path)
    else:
        birch = Birch(radius)
    if os.path.isfile(reduction_model_path):
        reduction_model = IPCA.from_pickle(reduction_model_path)
    else:
        reduction_model = IPCA(5)
    reduced_data = reduction_model.add_transform(dataframe.values)
    birch.add_pandas_data_frame(pd.DataFrame(reduced_data, index=dataframe.index.values))
    birch.to_files(full_path)
    birch.to_pickle(full_path)
    reduction_model.to_pickle(reduction_model)



