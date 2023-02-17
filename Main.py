import Constant
from Consensus import Consensus
from STAT_Feature_Engineering_Single import STAT_Feature_Engineering_Single
from TOP_Feature_Engineering_Extend import TOP_Feature_Engineering_Extend
from TOP_Feature_Engineering_Cut import TOP_Feature_Engineering_Cut
from Data_Save import DataSave
from Dataset import Dataset


def generate_top_dataset():
    dataset = Dataset()
    dataset.access_dataset(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_OFF)
    DataSave.save_gs_top_features_to_storage(dataset)
    DataSave.save_cs_top_features_to_storage(dataset)

    feature_engineering_cut = TOP_Feature_Engineering_Cut()
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
                               testing_normal_cut_data_dict, testing_attack_cut_data_dict, Constant.CUT_TOP_DATASET_PATH)

    # feature extend
    feature_engineering_extend = TOP_Feature_Engineering_Extend(cs_training_normal_data_dict, cs_training_attack_data_dict,
                                                                cs_testing_normal_data_dict, cs_testing_attack_data_dict)

    training_normal_feature_dict, training_attack_feature_dict, testing_normal_feature_dict, \
        testing_attack_feature_dict = feature_engineering_extend.get_extended_features()

    DataSave.save_top_features(training_normal_feature_dict, training_attack_feature_dict, testing_normal_feature_dict,
                               testing_attack_feature_dict, Constant.EXTENDED_TOP_DATASET_PATH)


def generate_stat_dataset():
    dataset = Dataset()
    dataset.access_dataset(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_OFF)

    stat_feature_engineering_single = STAT_Feature_Engineering_Single(dataset)
    stat_feature_engineering_single.parsing_dataset('instructions')


if __name__ == '__main__':
    print('Simulation Start')
    # generate_top_dataset()
    # consensus = Consensus(Constant.CUT_TOP_DATASET_PATH)
    # consensus.knn()
    # consensus.k_means()
    #
    # consensus = Consensus(Constant.EXTENDED_TOP_DATASET_PATH)
    # consensus.knn()
    # consensus.k_means()

    generate_stat_dataset()

    print('Simulation End')
