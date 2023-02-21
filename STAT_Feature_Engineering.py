import csv
import os
import numpy as np
import Constant


class STAT_Feature_Engineering:
    __row_size = 0
    __col_size = 0

    @classmethod
    def __get_min_feature_len(cls, feature_type, cs_stat_file_list):
        len_list = []
        feature_list = []
        for cs_id_feature_list in cs_stat_file_list:
            # cs_stat_cycle_list, cs_stat_instruction_list, cs_stat_branch_list
            for index, sub_cs_id_feature_list in enumerate(cs_id_feature_list):
                len_list.append(len(sub_cs_id_feature_list))
                if Constant.LIST_SEQUENCE[index] == feature_type:
                    int_cs_id_feature_list = []
                    for feature in sub_cs_id_feature_list:
                        int_cs_id_feature_list.append(int(feature.replace(',', '')))
                    feature_list.append(int_cs_id_feature_list)

        len_list = sorted(len_list)
        min_len = len_list[0]

        return min_len, feature_list

    def parsing_dataset(self, feature_type, cs_stat_file_list):
        min_len, feature_list = self.__get_min_feature_len(feature_type, cs_stat_file_list)
        temp_feature_list = []
        for single_feature in feature_list:
            temp_feature_list.append(single_feature[:min_len])

        self.__col_size = min_len
        self.__row_size = len(temp_feature_list)
        combined_array = np.empty(shape=(self.__row_size,), dtype=int)
        for index in range(0, min_len):
            record_list = []
            for sub_feature_list in temp_feature_list:
                data = sub_feature_list[index]
                record_list.append(data)
            record_array = np.array(record_list)
            combined_array = np.vstack((combined_array, record_array))

        combined_array = combined_array[1:]

        return combined_array

    @classmethod
    def get_combined_mixed_labeled_feature_list(cls, feature_array, feature_type):
        feature_list = []
        for feature in feature_array:
            temp_list = feature.tolist()
            if feature_type == Constant.NORMAL:
                temp_list.append(Constant.NORMAL_LABEL)
            elif feature_type == Constant.ATTACK:
                temp_list.append(Constant.ATTACK_LABEL)

            feature_list.append(temp_list)

        return feature_list

    @classmethod
    def get_randomly_mixed_feature_array(cls, normal_feature_list, attack_feature_list):
        total_feature_list = []
        for normal_feature in normal_feature_list:
            total_feature_list.append(normal_feature)

        for attack_feature in attack_feature_list:
            total_feature_list.append(attack_feature)

        np.random.shuffle(total_feature_list)

        total_label_list = []
        label_index = len(total_feature_list[0]) - 1
        for feature in total_feature_list:
            label = feature.pop(label_index)
            total_label_list.append(label)

        return np.array(total_feature_list), np.array(total_label_list)

    @classmethod
    def divide_training_and_testing_feature_array(cls, feature_array, label_array):
        total_size = len(label_array)
        training_size = round(total_size * 0.75)
        testing_size = round(total_size * 0.25)

        training_feature_array = feature_array[:training_size]
        testing_feature_array = feature_array[:testing_size]
        training_label_array = label_array[:training_size]
        testing_label_array = label_array[:testing_size]

        return training_feature_array, testing_feature_array, training_label_array, testing_label_array

    @classmethod
    def __read_csv(cls, path):
        read_line = []
        with open(path, 'r') as f:
            rdr = csv.reader(f)
            for line in rdr:
                read_line.append(line)

        return read_line

    @classmethod
    def __get_file_names(cls, path):
        file_name_list = []
        _file_name_list = os.listdir(path)
        for file_name in _file_name_list:
            if file_name.find('.csv') > 0:
                file_name_list.append(file_name)

        return file_name_list

    @classmethod
    def get_multiple_feature_and_label_array(cls, root_path):
        file_name_list = cls.__get_file_names(root_path)

        training_total_cycles_feature_list = []
        training_total_instructions_feature_list = []
        training_total_total_branch_feature_list = []
        training_total_cycles_label_list = []
        training_total_instructions_label_list = []
        training_total_branch_label_list = []

        testing_total_cycles_feature_list = []
        testing_total_instructions_feature_list = []
        testing_total_branch_feature_list = []
        testing_total_cycles_label_list = []
        testing_total_instructions_label_list = []
        testing_total_branch_label_list = []

        for file_name in file_name_list:
            split_file_name_list = file_name.split('_')
            if split_file_name_list[0] == 'training':
                if split_file_name_list[3] == 'feature.csv':
                    if split_file_name_list[2] == Constant.LIST_SEQUENCE[0]:  # cycles
                        cycles_feature_list = cls.__read_csv(root_path + '/' + file_name)
                        training_total_cycles_feature_list = cycles_feature_list[1:]
                    elif split_file_name_list[2] == Constant.LIST_SEQUENCE[1]:  # instructions
                        instructions_feature_list = cls.__read_csv(root_path + '/' + file_name)
                        training_total_instructions_feature_list = instructions_feature_list[1:]
                    elif split_file_name_list[2] == Constant.LIST_SEQUENCE[2]:  # branch
                        branch_feature_list = cls.__read_csv(root_path + '/' + file_name)
                        training_total_total_branch_feature_list = branch_feature_list[1:]
                elif split_file_name_list[3] == 'label.csv':
                    if split_file_name_list[2] == Constant.LIST_SEQUENCE[0]:  # cycles
                        cycles_label_list = cls.__read_csv(root_path + '/' + file_name)
                        training_total_cycles_label_list = cycles_label_list[1:]
                    elif split_file_name_list[2] == Constant.LIST_SEQUENCE[1]:  # instructions
                        instructions_label_list = cls.__read_csv(root_path + '/' + file_name)
                        training_total_instructions_label_list = instructions_label_list[1:]
                    elif split_file_name_list[2] == Constant.LIST_SEQUENCE[2]:  # branch
                        branch_label_list = cls.__read_csv(root_path + '/' + file_name)
                        training_total_branch_label_list = branch_label_list[1:]
            if split_file_name_list[0] == 'testing':
                if split_file_name_list[3] == 'feature.csv':
                    if split_file_name_list[2] == Constant.LIST_SEQUENCE[0]:  # cycles
                        cycles_feature_list = cls.__read_csv(root_path + '/' + file_name)
                        testing_total_cycles_feature_list = cycles_feature_list[1:]
                    elif split_file_name_list[2] == Constant.LIST_SEQUENCE[1]:  # instructions
                        instructions_feature_list = cls.__read_csv(root_path + '/' + file_name)
                        testing_total_instructions_feature_list = instructions_feature_list[1:]
                    elif split_file_name_list[2] == Constant.LIST_SEQUENCE[2]:  # branch
                        branch_feature_list = cls.__read_csv(root_path + '/' + file_name)
                        testing_total_branch_feature_list = branch_feature_list[1:]
                elif split_file_name_list[3] == 'label.csv':
                    if split_file_name_list[2] == Constant.LIST_SEQUENCE[0]:  # cycles
                        cycles_label_list = cls.__read_csv(root_path + '/' + file_name)
                        testing_total_cycles_label_list = cycles_label_list[1:]
                    elif split_file_name_list[2] == Constant.LIST_SEQUENCE[1]:  # instructions
                        instructions_label_list = cls.__read_csv(root_path + '/' + file_name)
                        testing_total_instructions_label_list = instructions_label_list[1:]
                    elif split_file_name_list[2] == Constant.LIST_SEQUENCE[2]:  # branch
                        branch_label_list = cls.__read_csv(root_path + '/' + file_name)
                        testing_total_branch_label_list = branch_label_list[1:]

