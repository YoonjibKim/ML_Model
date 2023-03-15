import math
import random

import numpy as np
import Constant
from Dataset import Dataset


class GS_Implementation(Dataset):
    __training_top_cycle_feature_array = None
    __training_top_cycle_label_array = None
    __testing_top_cycle_feature_array = None
    __testing_top_cycle_label_array = None
    __training_top_instructions_feature_array = None
    __training_top_instructions_label_array = None
    __testing_top_instructions_feature_array = None
    __testing_top_instructions_label_array = None
    __training_top_branch_feature_array = None
    __training_top_branch_label_array = None
    __testing_top_branch_feature_array = None
    __testing_top_branch_label_array = None

    def __init__(self, attack_scenario):
        self.__dataset = super()
        self.__dataset.access_dataset(attack_scenario[0], attack_scenario[1], attack_scenario[2])

    def get_top_cycle_feature_and_label_array(self):
        return self.__training_top_cycle_feature_array, self.__training_top_cycle_label_array, \
            self.__testing_top_cycle_feature_array, self.__testing_top_cycle_label_array

    def get_top_instructions_feature_and_label_array(self):
        return self.__training_top_instructions_feature_array, self.__training_top_instructions_label_array, \
            self.__testing_top_instructions_feature_array, self.__testing_top_instructions_label_array

    def get_top_branch_feature_and_label_array(self):
        return self.__training_top_branch_feature_array, self.__training_top_branch_label_array, \
            self.__testing_top_branch_feature_array, self.__testing_top_branch_label_array

    @classmethod
    def __get_extract_significant_symbol_list(cls, record_list):
        overhead_list = list(float(record[0][:-1]) for record in record_list)
        overhead_mean = np.mean(overhead_list)
        overhead_std = np.std(overhead_list)
        boundary = overhead_mean + overhead_std

        extracted_record_list = []
        for record in record_list:
            percent = float(record[0][:-1])
            if percent > boundary:
                symbol_name = record[1]
                extracted_record_list.append(symbol_name)

        return extracted_record_list

    def __get_norm_mean_std_record_list(self):
        temp_list = self.__dataset.get_attack_gs_record_file_list()
        attack_cycle_record_list = temp_list[Constant.CYCLE_INDEX]
        attack_instructions_record_list = temp_list[Constant.INSTRUCTIONS_INDEX]
        attack_branch_record_list = temp_list[Constant.BRANCH_INDEX]

        temp_list = self.__dataset.get_normal_gs_record_file_list()
        normal_cycle_record_list = temp_list[Constant.CYCLE_INDEX]
        normal_instructions_record_list = temp_list[Constant.INSTRUCTIONS_INDEX]
        normal_branch_record_list = temp_list[Constant.BRANCH_INDEX]

        attack_cycle_extracted_record_list = self.__get_extract_significant_symbol_list(attack_cycle_record_list)
        attack_instructions_extracted_record_list = \
            self.__get_extract_significant_symbol_list(attack_instructions_record_list)
        attack_branch_extracted_record_list = self.__get_extract_significant_symbol_list(attack_branch_record_list)

        normal_cycle_extracted_record_list = self.__get_extract_significant_symbol_list(normal_cycle_record_list)
        normal_instructions_extracted_record_list = \
            self.__get_extract_significant_symbol_list(normal_instructions_record_list)
        normal_branch_extracted_record_list = self.__get_extract_significant_symbol_list(normal_branch_record_list)

        cycle_extracted_record_set = set(attack_cycle_extracted_record_list) & set(normal_cycle_extracted_record_list)
        instruction_extracted_record_set = \
            set(attack_instructions_extracted_record_list) & set(normal_instructions_extracted_record_list)
        branch_extracted_record_set = \
            set(attack_branch_extracted_record_list) & set(normal_branch_extracted_record_list)

        return cycle_extracted_record_set, instruction_extracted_record_set, branch_extracted_record_set

    @classmethod
    def __get_intersection_feature_set(cls, norm_set, attack_top_list, normal_top_list):
        temp_attack_top_list = []
        for outer_list in attack_top_list:
            for record in outer_list:
                temp_list = record.split()
                temp_attack_top_list.append(temp_list[1])

        temp_normal_top_list = []
        for outer_list in normal_top_list:
            for record in outer_list:
                temp_list = record.split()
                temp_normal_top_list.append(temp_list[1])

        temp_intersection_top_set = set(temp_attack_top_list) & set(temp_normal_top_list)
        intersection_symbol_set = norm_set & temp_intersection_top_set

        return intersection_symbol_set

    @classmethod
    def __get_top_feature_list(cls, intersection_symbol_list, top_file_list):
        symbol_dict = {}
        for intersection_symbol_name in intersection_symbol_list:
            temp_list = []
            for outer_list in top_file_list:
                for temp_symbol_name in outer_list:
                    temp_symbol_list = temp_symbol_name.split()
                    top_symbol_name = temp_symbol_list[1]
                    if intersection_symbol_name == top_symbol_name:
                        top_overhead = temp_symbol_list[0][:-1]
                        top_overhead = float(top_overhead)
                        temp_list.append(top_overhead)
            symbol_dict[intersection_symbol_name] = temp_list

        return symbol_dict

    def top_feature_analysis(self):
        cycle_extracted_record_set, instructions_extracted_record_set, branch_extracted_record_set = \
            self.__get_norm_mean_std_record_list()

        temp_list = self.__dataset.get_attack_gs_top_file_list()
        attack_cycle_top_file_list = temp_list[Constant.CYCLE_INDEX]
        attack_instructions_top_file_list = temp_list[Constant.INSTRUCTIONS_INDEX]
        attack_branch_top_file_list = temp_list[Constant.BRANCH_INDEX]

        temp_list = self.__dataset.get_normal_gs_top_file_list()
        normal_cycle_top_file_list = temp_list[Constant.CYCLE_INDEX]
        normal_instructions_top_file_list = temp_list[Constant.INSTRUCTIONS_INDEX]
        normal_branch_top_file_list = temp_list[Constant.BRANCH_INDEX]

        intersection_cycle_symbol_set = self.__get_intersection_feature_set(cycle_extracted_record_set,
                                                                            attack_cycle_top_file_list,
                                                                            normal_cycle_top_file_list)
        intersection_instructions_symbol_set = self.__get_intersection_feature_set(instructions_extracted_record_set,
                                                                                   attack_instructions_top_file_list,
                                                                                   normal_instructions_top_file_list)
        intersection_branch_symbol_set = self.__get_intersection_feature_set(branch_extracted_record_set,
                                                                             attack_branch_top_file_list,
                                                                             normal_branch_top_file_list)

        attack_cycle_symbol_dict = self.__get_top_feature_list(intersection_cycle_symbol_set,
                                                               attack_cycle_top_file_list)
        normal_cycle_symbol_dict = self.__get_top_feature_list(intersection_cycle_symbol_set,
                                                               normal_cycle_top_file_list)
        attack_instructions_symbol_dict = self.__get_top_feature_list(intersection_instructions_symbol_set,
                                                                      attack_instructions_top_file_list)
        normal_instructions_symbol_dict = self.__get_top_feature_list(intersection_instructions_symbol_set,
                                                                      normal_instructions_top_file_list)
        attack_branch_symbol_dict = self.__get_top_feature_list(intersection_branch_symbol_set,
                                                                attack_branch_top_file_list)
        normal_branch_symbol_dict = self.__get_top_feature_list(intersection_branch_symbol_set,
                                                                normal_branch_top_file_list)

        attack_cycle_smallest_size, normal_cycle_smallest_size = \
            self.__calculate_smallest_feature_size(attack_cycle_symbol_dict, normal_cycle_symbol_dict)
        attack_instructions_smallest_size, normal_instructions_smallest_size = \
            self.__calculate_smallest_feature_size(attack_instructions_symbol_dict, normal_instructions_symbol_dict)
        attack_branch_smallest_size, normal_branch_smallest_size = \
            self.__calculate_smallest_feature_size(attack_branch_symbol_dict, normal_branch_symbol_dict)

        shrunk_attack_cycle_symbol_dict = \
            self.set_feature_size_to_smallest_one_dict(attack_cycle_smallest_size, attack_cycle_symbol_dict)
        shrunk_normal_cycle_symbol_dict = \
            self.set_feature_size_to_smallest_one_dict(normal_cycle_smallest_size, normal_cycle_symbol_dict)
        shrunk_attack_instructions_symbol_dict = \
            self.set_feature_size_to_smallest_one_dict(attack_instructions_smallest_size,
                                                       attack_instructions_symbol_dict)
        shrunk_normal_instructions_symbol_dict = \
            self.set_feature_size_to_smallest_one_dict(normal_instructions_smallest_size,
                                                       normal_instructions_symbol_dict)
        shrunk_attack_branch_symbol_dict = \
            self.set_feature_size_to_smallest_one_dict(attack_branch_smallest_size, attack_branch_symbol_dict)
        shrunk_normal_branch_symbol_dict = \
            self.set_feature_size_to_smallest_one_dict(normal_branch_smallest_size, normal_branch_symbol_dict)

        self.__training_top_cycle_feature_array, self.__training_top_cycle_label_array, \
            self.__testing_top_cycle_feature_array, self.__testing_top_cycle_label_array = \
            self.__get_training_and_testing_dataset_array(shrunk_attack_cycle_symbol_dict,
                                                          shrunk_normal_cycle_symbol_dict,
                                                          attack_cycle_smallest_size,
                                                          normal_cycle_smallest_size)
        self.__training_top_instructions_feature_array, self.__training_top_instructions_label_array, \
            self.__testing_top_instructions_feature_array, self.__testing_top_instructions_label_array = \
            self.__get_training_and_testing_dataset_array(shrunk_attack_instructions_symbol_dict,
                                                          shrunk_normal_instructions_symbol_dict,
                                                          attack_instructions_smallest_size,
                                                          normal_instructions_smallest_size)
        self.__training_top_branch_feature_array, self.__training_top_branch_label_array, \
            self.__testing_top_branch_feature_array, self.__testing_top_branch_label_array = \
            self.__get_training_and_testing_dataset_array(shrunk_attack_branch_symbol_dict,
                                                          shrunk_normal_branch_symbol_dict,
                                                          attack_branch_smallest_size,
                                                          normal_branch_smallest_size)

    @classmethod
    def __get_training_and_testing_dataset_array(cls, attack_feature_dict, normal_feature_dict, attack_size,
                                                 normal_size):
        total_feature_list = []
        for attack_feature_name, attack_feature_list in attack_feature_dict.items():
            normal_feature_list = normal_feature_dict[attack_feature_name]
            normal_feature_list.extend(attack_feature_list)
            total_feature_list.append(normal_feature_list)

        attack_label_list = list(Constant.ATTACK_LABEL for i in range(0, attack_size))
        normal_label_list = list(Constant.NORMAL_LABEL for i in range(0, normal_size))
        normal_label_list.extend(attack_label_list)
        total_feature_list.append(normal_label_list)

        total_feature_array = np.array(total_feature_list)
        total_feature_array = total_feature_array.T

        np.random.shuffle(total_feature_array)

        feature_list = []
        label_list = []
        for record in total_feature_array:
            label_list.append(record[len(record) - 1])
            feature_list.append(record[:-1])

        feature_array = np.array(feature_list)
        label_array = np.array(label_list)

        total_feature_size = attack_size + normal_size
        training_dataset_size = total_feature_size * Constant.TRAINING_FEATURE_RATIO
        training_dataset_size = round(training_dataset_size)
        testing_dataset_size = total_feature_size - training_dataset_size

        training_feature_array = feature_array[0:training_dataset_size]
        testing_feature_array = feature_array[training_dataset_size:training_dataset_size + testing_dataset_size]
        training_label_array = label_array[0:training_dataset_size]
        testing_label_array = label_array[training_dataset_size:training_dataset_size + testing_dataset_size]

        return training_feature_array, training_label_array, testing_feature_array, testing_label_array

    @classmethod
    def set_feature_size_to_smallest_one_dict(cls, smallest_size, symbol_dict):
        shrunk_symbol_dict = {}

        for feature_name, symbol_list in symbol_dict.items():
            large_list_size = len(symbol_list)
            result = large_list_size / smallest_size
            quotient = math.floor(result)
            remainder = result - quotient
            total_remainder = 0
            index = 0
            count = 0
            chosen_data_list = []
            while index < large_list_size:
                if index % quotient == 0:
                    if total_remainder > quotient:
                        chosen_data_list.append(symbol_list[index])
                        index += quotient
                        total_remainder = total_remainder - quotient
                    else:
                        total_remainder += remainder
                        chosen_data_list.append(symbol_list[index])
                    count += 1

                index += 1

            shrunk_index_size = len(chosen_data_list)
            index_list = list(index for index in range(0, shrunk_index_size))
            random_index_list = random.sample(index_list, smallest_size)
            random_index_list = sorted(random_index_list)

            final_shrunk_list = []
            for index in random_index_list:
                final_shrunk_list.append(chosen_data_list[index])

            shrunk_symbol_dict[feature_name] = final_shrunk_list

        return shrunk_symbol_dict

    @classmethod
    def __calculate_smallest_feature_size(cls, attack_symbol_dict, normal_symbol_dict):
        attack_temp_list = list(len(values) for values in attack_symbol_dict.values())
        attack_temp_list = sorted(attack_temp_list, reverse=False)

        normal_temp_list = list(len(values) for values in normal_symbol_dict.values())
        normal_temp_list = sorted(normal_temp_list, reverse=False)

        return attack_temp_list[0], normal_temp_list[0]

    def stat_feature_analysis(self):
        attack_gs_stat_list = self.get_attack_gs_stat_file_list()
        normal_gs_stat_list = self.get_normal_gs_stat_file_list()

        print(attack_gs_stat_list)
