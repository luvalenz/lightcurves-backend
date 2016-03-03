__author__ = 'lucas'

import yaml, os, glob
import pandas as pd
import numpy as np
from offline_algorithms import Birch, IncrementalPCA as IPCA


class OfflineInterface(object):

    __instance = None


    def __new__(cls):
        if cls.__instance == None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    @staticmethod
    def get_config(self):
        print(glob.glob('*'))
        return yaml.safe_load(open('config.yaml'))

    def __init__(self):
        #read config file
        config = self.get_config()
        clusters_root_path = config['clusters_path']
        data_path = config['data_path']
        self.reduction_algorithm = config['reduction_algorithm']
        self.dimensionality = config['dimensionality']
        self.clustering_algorithm = config['clustering_algorithm']
        self.clustering_parameters = config['clustering_parameters']
        #create directory
        dim_red_folder_name = '{0}_{1}'.format(self.reduction_algorithm, self.dimensionality)
        clust_algo_folder_name = self.clustering_algorithm
        for param in self.clustering_parameters:
            clust_algo_folder_name += '_' + str(param)
        self.clusters_full_path = os.path.join(clusters_root_path, dim_red_folder_name, clust_algo_folder_name)
        if not os.path.exists(self.clusters_full_path):
            try:
                os.makedirs(self.clusters_full_path)
            except OSError:
                print("Directory already exists")

    def get_clustering_model(self):
        if self.clustering_algorithm == 'birch':
            Model = Birch
        clustering_model_path = os.path.join(self.clusters_full_path, 'clustering_model.pkl')
        if os.path.isfile(clustering_model_path):
            model = Model.from_pickle(clustering_model_path)
        else:
            model = Model(**self.clustering_parameters)
        return model


    def get_reduction_model(self):
        if self.reduction_algorithm == 'ipca':
            Model = IPCA
        clustering_model_path = os.path.join(self.clusters_full_path, 'reduction_model.pkl')
        if os.path.isfile(clustering_model_path):
            model = Model.from_pickle(clustering_model_path)
        else:
            model = Model(self.dimensionality)
        return model


    def add_data(self, dataframe):
        clustering_model = self.get_clustering_model()
        reduction_model = self.get_reduction_model()
        reduced_data = reduction_model.add_transform(dataframe.values)
        clustering_model.add_pandas_data_frame(pd.DataFrame(reduced_data, index=dataframe.index.values))
        clustering_model._to_files(self.clusters_full_path)
        clustering_model.to_pickle(self.clusters_full_path, 'clustering_model')
        reduction_model.to_pickle(self.clusters_full_path, 'reduction_model')






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










