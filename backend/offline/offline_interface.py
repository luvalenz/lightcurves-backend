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
        return yaml.safe_load(open('config.json'))

    def __init__(self):
        #read config file
        self.config = self.get_config()

    def get_clustering_model(self, index):
        pass

    def get_reduction_model(self, index):
        pass

    def reduce_all(self, time_series_database_index):
        pass
        #update reduced model
        #recalculate ALL (important)

    def cluster_all(self, time_series_database_index, cluster_database, index):
        pass

