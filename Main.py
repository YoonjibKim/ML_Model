import Constant
from Consensus import Consensus
from Feature_Engineering_Extend import Feature_Engineering_Extend
from Feature_Engineering_Cut import Feature_Engineering_Cut
from Data_Save import DataSave
from Dataset import Dataset


def generate_dataset():
    dataset = Dataset()
    dataset.access_dataset(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_OFF)
    DataSave.save_gs_features_to_storage(dataset)
    DataSave.save_cs_features_to_storage(dataset)

    feature_engineering_cut = Feature_Engineering_Cut()
    feature_engineering_cut.parsing_dataset(dataset.get_base_dir_path())
    feature_engineering_cut.load_data_for_cs()
    cs_training_normal_data_dict, cs_training_attack_data_dict, cs_testing_normal_data_dict, \
        cs_testing_attack_data_dict, intersection_symbol_list = feature_engineering_cut.get_feature_for_cs()

    # feature cut
    training_normal_cut_data_dict, training_attack_cut_data_dict, testing_normal_cut_data_dict, \
        testing_attack_cut_data_dict = \
        feature_engineering_cut.get_cut_feature_for_cs(cs_training_normal_data_dict, cs_training_attack_data_dict,
                                                       cs_testing_normal_data_dict, cs_testing_attack_data_dict,
                                                       intersection_symbol_list)

    DataSave.save_top_features(training_normal_cut_data_dict, training_attack_cut_data_dict,
                               testing_normal_cut_data_dict, testing_attack_cut_data_dict, Constant.CUT_DATASET_PATH)

    # feature extend
    feature_engineering_extend = Feature_Engineering_Extend(cs_training_normal_data_dict, cs_training_attack_data_dict,
                                                            cs_testing_normal_data_dict, cs_testing_attack_data_dict)

    training_normal_feature_dict, training_attack_feature_dict, testing_normal_feature_dict, \
        testing_attack_feature_dict = feature_engineering_extend.get_extended_features()

    DataSave.save_top_features(training_normal_feature_dict, training_attack_feature_dict, testing_normal_feature_dict,
                               testing_attack_feature_dict, Constant.EXTENDED_DATASET_PATH)


if __name__ == '__main__':
    print('Simulation Start')
    # generate_dataset()

    consensus = Consensus(Constant.CUT_DATASET_PATH)
    consensus.knn()
    consensus.k_means()
    consensus = Consensus(Constant.EXTENDED_DATASET_PATH)
    consensus.knn()
    consensus.k_means()

    print('Simulation End')
