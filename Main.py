import Constant
from Consensus import Consensus
from Feature_Engineering import Feature_Engineering
from Data_Save import DataSave
from Dataset import Dataset

if __name__ == '__main__':
    dataset = Dataset()
    dataset.access_dataset(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_OFF)
    DataSave.save_gs_features_to_storage(dataset)
    DataSave.save_cs_features_to_storage(dataset)

    feature_engineering = Feature_Engineering()
    feature_engineering.parsing_dataset(dataset.get_base_dir_path())
    feature_engineering.load_data_for_cs()
    cs_training_normal_data_dict, cs_training_attack_data_dict, cs_testing_normal_data, cs_testing_attack_data = \
        feature_engineering.feature_configuration_setting_for_cs()

    consensus = Consensus(cs_training_normal_data_dict, cs_training_attack_data_dict, cs_testing_normal_data,
                          cs_testing_attack_data)
    consensus.knn()


