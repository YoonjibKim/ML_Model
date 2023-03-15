import numpy as np

import Constant
from Authentication_Time_Feature_Engineering import Authentication_Time_Feature_Engineering
from Combined_Feature_Engineering import Combined_Feature_Engineering
from Consensus import Consensus
from Data_Save import DataSave
from Dataset import Dataset
from STAT_Feature_Engineering import STAT_Feature_Engineering
from TOP_Feature_Engineering_Cut import TOP_Feature_Engineering_Cut
from TOP_Feature_Engineering_Extend import TOP_Feature_Engineering_Extend


class CS_Implementation:
    __training_stat_cycle_feature_array = None
    __training_stat_cycle_label_array = None
    __testing_stat_cycle_feature_array = None
    __testing_stat_cycle_label_array = None
    __training_stat_instructions_feature_array = None
    __training_stat_instructions_label_array = None
    __testing_stat_instructions_feature_array = None
    __testing_stat_instructions_label_array = None
    __training_stat_branch_feature_array = None
    __training_stat_branch_label_array = None
    __testing_stat_branch_feature_array = None
    __testing_stat_branch_label_array = None
    __training_stat_all_feature_array = None
    __training_stat_all_label_array = None
    __testing_stat_all_feature_array = None
    __testing_stat_all_label_array = None

    __training_time_diff_feature_array = None
    __training_time_diff_label_array = None
    __testing_time_diff_feature_array = None
    __testing_time_diff_label_array = None

    __training_stat_time_diff_feature_array = None
    __training_stat_time_diff_label_array = None
    __testing_stat_time_diff_feature_array = None
    __testing_stat_time_diff_label_array = None

    __training_top_stat_time_diff_feature_array = None
    __training_top_stat_time_diff_label_array = None
    __testing_top_stat_time_diff_feature_array = None
    __testing_top_stat_time_diff_label_array = None

    def __init__(self, attack_scenario):
        self.__attack_scenario = attack_scenario

    def get_shrunk_feature_array(self):
        dataset = Dataset()
        dataset.access_dataset(self.__attack_scenario[0], self.__attack_scenario[1], self.__attack_scenario[2])
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
                                   Constant.CS_CUT_TOP_DATASET_PATH)

        consensus = Consensus(Constant.CS_CUT_TOP_DATASET_PATH)
        training_feature_array, testing_feature_array, training_label_array, testing_label_array = \
            consensus.get_ml_training_and_testing_features()

        return training_feature_array, training_label_array, testing_feature_array, testing_label_array

    def get_extended_feature_array(self):
        dataset = Dataset()
        dataset.access_dataset(self.__attack_scenario[0], self.__attack_scenario[1], self.__attack_scenario[2])
        DataSave.save_gs_top_features_to_storage(dataset)
        DataSave.save_cs_top_features_to_storage(dataset)
        feature_engineering_cut = TOP_Feature_Engineering_Cut()
        feature_engineering_cut.parsing_dataset(dataset.get_base_dir_path())
        feature_engineering_cut.load_data_for_cs()
        cs_training_normal_data_dict, cs_training_attack_data_dict, cs_testing_normal_data_dict, \
            cs_testing_attack_data_dict, intersection_symbol_list = feature_engineering_cut.get_feature_for_cs()

        # feature extend
        feature_engineering_extend = TOP_Feature_Engineering_Extend(cs_training_normal_data_dict,
                                                                    cs_training_attack_data_dict,
                                                                    cs_testing_normal_data_dict,
                                                                    cs_testing_attack_data_dict)

        training_normal_feature_dict, training_attack_feature_dict, testing_normal_feature_dict, \
            testing_attack_feature_dict = feature_engineering_extend.get_extended_features()

        DataSave.save_top_features(training_normal_feature_dict, training_attack_feature_dict,
                                   testing_normal_feature_dict,
                                   testing_attack_feature_dict, Constant.CS_EXTENDED_TOP_DATASET_PATH)

        consensus = Consensus(Constant.CS_EXTENDED_TOP_DATASET_PATH)
        training_feature_array, testing_feature_array, training_label_array, testing_label_array = \
            consensus.get_ml_training_and_testing_features()

        return training_feature_array, training_label_array, testing_feature_array, testing_label_array

    def stat_feature_analysis(self):
        chosen_feature_list = Constant.LIST_SEQUENCE
        for feature_type in chosen_feature_list:
            self.__save_raw_stat_dataset(feature_type, self.__attack_scenario)

        self.__training_stat_all_feature_array, self.__training_stat_all_label_array, \
            self.__testing_stat_all_feature_array, self.__testing_stat_all_label_array = \
            self.__extract_raw_stat_dataset(chosen_feature_list)

        for feature_list in chosen_feature_list:
            if feature_list == Constant.LIST_SEQUENCE[0]:  # cycle
                self.__training_stat_cycle_feature_array, self.__training_stat_cycle_label_array, \
                    self.__testing_stat_cycle_feature_array, self.__testing_stat_cycle_label_array = \
                    self.__extract_raw_stat_dataset([feature_list])
            elif feature_list == Constant.LIST_SEQUENCE[1]:  # instructions
                self.__training_stat_instructions_feature_array, self.__training_stat_instructions_label_array, \
                    self.__testing_stat_instructions_feature_array, self.__testing_stat_instructions_label_array = \
                    self.__extract_raw_stat_dataset([feature_list])
            elif feature_list == Constant.LIST_SEQUENCE[2]:  # branch
                self.__training_stat_branch_feature_array, self.__training_stat_branch_label_array, \
                    self.__testing_stat_branch_feature_array, self.__testing_stat_branch_label_array = \
                    self.__extract_raw_stat_dataset([feature_list])
        # ---------------------------------------------------------------------------------------------------------
        authentication_time_feature_engineering = Authentication_Time_Feature_Engineering(self.__attack_scenario[0],
                                                                                          self.__attack_scenario[1],
                                                                                          self.__attack_scenario[2])

        training_feature_array, self.__training_time_diff_label_array, \
            testing_feature_array, self.__testing_time_diff_label_array = \
            authentication_time_feature_engineering.divide_training_and_testing_features()

        self.__training_time_diff_feature_array = training_feature_array.reshape(-1, 1)
        self.__testing_time_diff_feature_array = testing_feature_array.reshape(-1, 1)
        # ---------------------------------------------------------------------------------------------------------
        chosen_feature_list = Constant.LIST_SEQUENCE
        stat_mixed_list = \
            STAT_Feature_Engineering.get_feature_and_label_list(Constant.RAW_STAT_DATASET_PATH, chosen_feature_list)
        authentication_time_feature_engineering = Authentication_Time_Feature_Engineering(self.__attack_scenario[0],
                                                                                          self.__attack_scenario[1],
                                                                                          self.__attack_scenario[2])
        time_diff_mixed_list = authentication_time_feature_engineering.get_feature_and_label_list()

        combined_feature_engineering = Combined_Feature_Engineering()
        stat_attack_list, stat_normal_list = \
            combined_feature_engineering.divide_attack_and_normal_features(stat_mixed_list)
        time_diff_attack_list, time_diff_normal_list = \
            combined_feature_engineering.divide_attack_and_normal_features(time_diff_mixed_list)

        stat_same_length_attack_list, time_diff_same_length_attack_list = \
            combined_feature_engineering.make_same_feature_length(stat_attack_list, time_diff_attack_list)
        stat_same_length_normal_list, time_diff_same_length_normal_list = \
            combined_feature_engineering.make_same_feature_length(stat_normal_list, time_diff_normal_list)

        self.__training_stat_time_diff_feature_array, self.__training_stat_time_diff_label_array, \
            self.__testing_stat_time_diff_feature_array, self.__testing_stat_time_diff_label_array = \
            combined_feature_engineering.get_training_and_testing_feature_array(stat_same_length_attack_list,
                                                                                time_diff_same_length_attack_list,
                                                                                stat_same_length_normal_list,
                                                                                time_diff_same_length_normal_list)
        # ---------------------------------------------------------------------------------------------------------
        chosen_feature_list = Constant.LIST_SEQUENCE
        stat_mixed_list = \
            STAT_Feature_Engineering.get_feature_and_label_list(Constant.RAW_STAT_DATASET_PATH, chosen_feature_list)
        authentication_time_feature_engineering = \
            Authentication_Time_Feature_Engineering(self.__attack_scenario[0], self.__attack_scenario[1],
                                                    self.__attack_scenario[2])
        time_diff_mixed_list = authentication_time_feature_engineering.get_feature_and_label_list()

        combined_feature_engineering = Combined_Feature_Engineering()
        stat_attack_list, stat_normal_list = \
            combined_feature_engineering.divide_attack_and_normal_features(stat_mixed_list)
        _time_diff_attack_list, _time_diff_normal_list = \
            combined_feature_engineering.divide_attack_and_normal_features(time_diff_mixed_list)

        consensus = Consensus(Constant.CS_CUT_TOP_DATASET_PATH)
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

        size_list = [len(top_same_length_normal_list), len(stat_same_length_normal_list),
                     len(stat_same_length_normal_list)]
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

        self.__training_top_stat_time_diff_feature_array, self.__training_top_stat_time_diff_label_array, \
            self.__testing_top_stat_time_diff_feature_array, self.__testing_top_stat_time_diff_label_array = \
            combined_feature_engineering.get_training_and_testing_feature_array(top_stat_attack_list,
                                                                                time_diff_attack_list,
                                                                                top_stat_normal_list,
                                                                                time_diff_normal_list)

    def get_stat_cycle_feature_and_label_array(self):
        return self.__training_stat_cycle_feature_array, self.__training_stat_cycle_label_array, \
            self.__testing_stat_cycle_feature_array, self.__testing_stat_cycle_label_array

    def get_stat_instructions_feature_and_label_array(self):
        return self.__training_stat_instructions_feature_array, self.__training_stat_instructions_label_array, \
            self.__testing_stat_instructions_feature_array, self.__testing_stat_instructions_label_array

    def get_stat_branch_feature_and_label_array(self):
        return self.__training_stat_branch_feature_array, self.__training_stat_branch_label_array, \
            self.__testing_stat_branch_feature_array, self.__testing_stat_branch_label_array

    def get_stat_all_feature_and_label_array(self):
        return self.__training_stat_all_feature_array, self.__training_stat_all_label_array, \
            self.__testing_stat_all_feature_array, self.__testing_stat_all_label_array

    def get_time_diff_feature_and_label_array(self):
        return self.__training_time_diff_feature_array, self.__training_time_diff_label_array, \
            self.__testing_time_diff_feature_array, self.__testing_time_diff_label_array

    def get_stat_time_diff_feature_and_label_array(self):
        return self.__training_stat_time_diff_feature_array, self.__training_stat_time_diff_label_array, \
            self.__testing_stat_time_diff_feature_array, self.__testing_stat_time_diff_label_array

    def get_top_stat_time_diff_feature_and_label_array(self):
        return self.__training_top_stat_time_diff_feature_array, self.__training_top_stat_time_diff_label_array, \
            self.__testing_top_stat_time_diff_feature_array, self.__testing_top_stat_time_diff_label_array

    @classmethod
    def __save_raw_stat_dataset(cls, feature_type, attack_scenario):
        dataset = Dataset()
        dataset.access_dataset(attack_scenario[0], attack_scenario[1], attack_scenario[2])

        stat_feature_engineering_single = STAT_Feature_Engineering()
        normal_data_array = \
            stat_feature_engineering_single.parsing_dataset(feature_type, dataset.get_normal_cs_stat_file_list())

        attack_data_array = \
            stat_feature_engineering_single.parsing_dataset(feature_type, dataset.get_attack_cs_stat_file_list())

        normal_labeled_feature_list = \
            stat_feature_engineering_single.get_combined_mixed_labeled_feature_list(normal_data_array, Constant.NORMAL)
        attack_labeled_feature_list = \
            stat_feature_engineering_single.get_combined_mixed_labeled_feature_list(attack_data_array, Constant.ATTACK)

        total_feature_array, total_label_array = \
            stat_feature_engineering_single.get_mixed_feature_array(normal_labeled_feature_list,
                                                                    attack_labeled_feature_list)

        DataSave.save_stat_raw_feature(total_feature_array, total_label_array, feature_type)

    @classmethod
    def __extract_raw_stat_dataset(cls, chosen_feature_list):
        combined_feature_label_list = \
            STAT_Feature_Engineering.get_feature_and_label_list(Constant.RAW_STAT_DATASET_PATH, chosen_feature_list)

        training_feature_array, training_label_array, testing_feature_array, testing_label_array = \
            STAT_Feature_Engineering.divide_training_and_testing_features(combined_feature_label_list)

        DataSave.save_stat_processed_feature(training_feature_array, training_label_array, testing_feature_array,
                                             testing_label_array)

        return training_feature_array, training_label_array, testing_feature_array, testing_label_array
