from backend.offline.offline_interface import OfflineInterface

if __name__ == "__main__":
    interface = OfflineInterface()
    ts_db = interface.get_time_series_database()
    #transfer data
    #interface.transfer_time_series('macho', 1,0)
    #calculate features
    #interface.calculate_missing_features(0, 5)
    #interface.recalculate_all_features(0, 5)
    #setup
    interface.setup()
    #reduce dimensionality
    #interface.reduce_all(0, 0, 0)
    #cluster
    interface.cluster_all(0, 0, 0)
