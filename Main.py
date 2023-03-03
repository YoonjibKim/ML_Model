import numpy as np

import Constant
import Data_Save
from Combined_Feature_Engineering import Combined_Feature_Engineering
from Consensus import Consensus
from STAT_Feature_Engineering import STAT_Feature_Engineering
from TOP_Feature_Engineering_Extend import TOP_Feature_Engineering_Extend
from TOP_Feature_Engineering_Cut import TOP_Feature_Engineering_Cut
from Authentication_Time_Feature_Engineering import Authentication_Time_Feature_Engineering
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
                               testing_normal_cut_data_dict, testing_attack_cut_data_dict,
                               Constant.CUT_TOP_DATASET_PATH)

    # feature extend
    feature_engineering_extend = TOP_Feature_Engineering_Extend(cs_training_normal_data_dict,
                                                                cs_training_attack_data_dict,
                                                                cs_testing_normal_data_dict,
                                                                cs_testing_attack_data_dict)

    training_normal_feature_dict, training_attack_feature_dict, testing_normal_feature_dict, \
        testing_attack_feature_dict = feature_engineering_extend.get_extended_features()

    DataSave.save_top_features(training_normal_feature_dict, training_attack_feature_dict, testing_normal_feature_dict,
                               testing_attack_feature_dict, Constant.EXTENDED_TOP_DATASET_PATH)


def save_raw_stat_dataset(param_feature_type):
    dataset = Dataset()
    dataset.access_dataset(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_ON)

    stat_feature_engineering_single = STAT_Feature_Engineering()
    normal_data_array = \
        stat_feature_engineering_single.parsing_dataset(param_feature_type, dataset.get_normal_cs_stat_file_list())

    attack_data_array = \
        stat_feature_engineering_single.parsing_dataset(param_feature_type, dataset.get_attack_cs_stat_file_list())

    normal_labeled_feature_list = \
        stat_feature_engineering_single.get_combined_mixed_labeled_feature_list(normal_data_array,
                                                                                Constant.NORMAL)
    attack_labeled_feature_list = \
        stat_feature_engineering_single.get_combined_mixed_labeled_feature_list(attack_data_array,
                                                                                Constant.ATTACK)

    total_feature_array, total_label_array = \
        stat_feature_engineering_single.get_mixed_feature_array(normal_labeled_feature_list,
                                                                attack_labeled_feature_list)

    DataSave.save_stat_raw_feature(total_feature_array, total_label_array, param_feature_type)


def extract_raw_stat_dataset(param_chosen_feature_list):
    combined_feature_label_list = \
        STAT_Feature_Engineering.get_feature_and_label_list(Constant.RAW_STAT_DATASET_PATH, param_chosen_feature_list)

    ret_training_feature_array, ret_training_label_array, ret_testing_feature_array, ret_testing_label_array = \
        STAT_Feature_Engineering.divide_training_and_testing_features(combined_feature_label_list)

    DataSave.save_stat_processed_feature(ret_training_feature_array, ret_training_label_array,
                                         ret_testing_feature_array, ret_testing_label_array)

    return ret_training_feature_array, ret_training_label_array, ret_testing_feature_array, ret_testing_label_array


if __name__ == '__main__':
    print('Simulation Start')
    # generate_top_dataset()

    # --------------------------------- top ml: cut ---------------------------------
    # consensus = Consensus(Constant.CUT_TOP_DATASET_PATH)
    # training_feature_array, testing_feature_array, training_label_array, testing_label_array = \
    #     consensus.get_ml_features()

    # --------------------------------- top ml: extend ---------------------------------
    # consensus = Consensus(Constant.EXTENDED_TOP_DATASET_PATH)
    # training_feature_array, testing_feature_array, training_label_array, testing_label_array = \
    #     consensus.get_ml_features()

    # --------------------------------- stat ml ---------------------------------
    # chosen_feature_list = Constant.LIST_SEQUENCE
    # chosen_feature_list = [Constant.LIST_SEQUENCE[2]]
    # for feature_type in chosen_feature_list:
    #     save_raw_stat_dataset(feature_type)
    #
    # training_feature_array, training_label_array, testing_feature_array, testing_label_array = \
    #     extract_raw_stat_dataset(chosen_feature_list)

    # --------------------------------- time diff ml (one feature) ---------------------------------
    # authentication_time_feature_engineering = \
    #     Authentication_Time_Feature_Engineering(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_ON)
    #
    # training_feature_array, training_label_array, testing_feature_array, testing_label_array = \
    #     authentication_time_feature_engineering.divide_training_and_testing_features()
    #
    # training_feature_array = training_feature_array.reshape(-1, 1)
    # testing_feature_array = testing_feature_array.reshape(-1, 1)

    # --------------------------------- stat and time diff ml ---------------------------------
    # chosen_feature_list = [Constant.LIST_SEQUENCE[2]]
    # chosen_feature_list = Constant.LIST_SEQUENCE
    # stat_mixed_list = \
    #     STAT_Feature_Engineering.get_feature_and_label_list(Constant.RAW_STAT_DATASET_PATH, chosen_feature_list)
    # authentication_time_feature_engineering = \
    #     Authentication_Time_Feature_Engineering(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_ON)
    # time_diff_mixed_list = authentication_time_feature_engineering.get_feature_and_label_list()
    #
    # combined_feature_engineering = Combined_Feature_Engineering()
    # stat_attack_list, stat_normal_list = combined_feature_engineering.divide_attack_and_normal_features(stat_mixed_list)
    # time_diff_attack_list, time_diff_normal_list = \
    #     combined_feature_engineering.divide_attack_and_normal_features(time_diff_mixed_list)
    #
    # stat_same_length_attack_list, time_diff_same_length_attack_list = \
    #     combined_feature_engineering.make_same_feature_length(stat_attack_list, time_diff_attack_list)
    # stat_same_length_normal_list, time_diff_same_length_normal_list = \
    #     combined_feature_engineering.make_same_feature_length(stat_normal_list, time_diff_normal_list)
    #
    # training_feature_array, training_label_array, testing_feature_array, testing_label_array = \
    #     combined_feature_engineering.get_training_and_testing_feature_array(stat_same_length_attack_list,
    #                                                                         time_diff_same_length_attack_list,
    #                                                                         stat_same_length_normal_list,
    #                                                                         time_diff_same_length_normal_list)

    # --------------------------------- total feature combinations ---------------------------------
    # chosen_feature_list = [Constant.LIST_SEQUENCE[2]]
    chosen_feature_list = Constant.LIST_SEQUENCE
    stat_mixed_list = \
        STAT_Feature_Engineering.get_feature_and_label_list(Constant.RAW_STAT_DATASET_PATH, chosen_feature_list)
    authentication_time_feature_engineering = \
        Authentication_Time_Feature_Engineering(Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_ON)
    time_diff_mixed_list = authentication_time_feature_engineering.get_feature_and_label_list()

    combined_feature_engineering = Combined_Feature_Engineering()
    stat_attack_list, stat_normal_list = combined_feature_engineering.divide_attack_and_normal_features(stat_mixed_list)
    _time_diff_attack_list, _time_diff_normal_list = \
        combined_feature_engineering.divide_attack_and_normal_features(time_diff_mixed_list)

    consensus = Consensus(Constant.CUT_TOP_DATASET_PATH)
    top_attack_feature_list, top_normal_feature_list = consensus.divide_attack_and_normal_features()

    _top_same_length_normal_list, time_diff_same_length_normal_list = \
        combined_feature_engineering.make_same_feature_length(top_normal_feature_list, _time_diff_normal_list)
    _top_same_length_attack_list, time_diff_same_length_attack_list = \
        combined_feature_engineering.make_same_feature_length(top_attack_feature_list, _time_diff_attack_list)

    top_same_length_normal_list, stat_same_length_normal_list = \
        combined_feature_engineering.make_same_feature_length(top_normal_feature_list,
                                                              stat_normal_list)
    top_same_length_attack_list, stat_same_length_attack_list = \
        combined_feature_engineering.make_same_feature_length(top_attack_feature_list,
                                                              stat_attack_list)

    size_list = [len(top_same_length_normal_list), len(stat_same_length_normal_list), len(stat_same_length_normal_list)]
    size_list = sorted(size_list)
    min_size = size_list[0]

    top_attack_list = top_same_length_attack_list[0:min_size]
    top_normal_list = top_same_length_normal_list[0:min_size]
    stat_attack_list = stat_same_length_attack_list[0:min_size]
    stat_normal_list = stat_same_length_normal_list[0:min_size]
    time_diff_attack_list = time_diff_same_length_attack_list[0:min_size]
    time_diff_normal_list = time_diff_same_length_normal_list[0:min_size]

    temp_top_attack_list = []
    for record in top_attack_list:
        temp_top_attack_list.append(record[:-1])
    temp_top_normal_list = []
    for record in top_normal_list:
        temp_top_normal_list.append(record[:-1])

    top_attack_array = np.array(temp_top_attack_list)
    top_normal_array = np.array(temp_top_normal_list)
    stat_attack_array = np.array(stat_attack_list)
    stat_normal_array = np.array(stat_normal_list)
    top_stat_attack_list = list(np.append(top_attack_array, stat_attack_array, axis=1))
    top_stat_normal_list = list(np.append(top_normal_array, stat_normal_array, axis=1))

    training_feature_array, training_label_array, testing_feature_array, testing_label_array = \
        combined_feature_engineering.get_training_and_testing_feature_array(top_stat_attack_list,
                                                                            time_diff_attack_list,
                                                                            top_stat_normal_list,
                                                                            time_diff_normal_list)

    # --------------------------------- ML ---------------------------------
    Consensus.knn(training_feature_array, training_label_array, testing_feature_array, testing_label_array)
    Consensus.k_means(testing_feature_array, testing_label_array)
    Consensus.dnn_run(training_feature_array, training_label_array, testing_feature_array, testing_label_array)

    print('Simulation End')
