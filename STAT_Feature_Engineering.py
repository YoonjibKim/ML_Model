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
    def get_mixed_feature_array(cls, normal_feature_list, attack_feature_list):
        total_feature_list = []
        for normal_feature in normal_feature_list:
            total_feature_list.append(normal_feature)

        for attack_feature in attack_feature_list:
            total_feature_list.append(attack_feature)

        total_label_list = []
        label_index = len(total_feature_list[0]) - 1
        for feature in total_feature_list:
            label = feature.pop(label_index)
            total_label_list.append(label)

        return np.array(total_feature_list), np.array(total_label_list)

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
    def get_feature_and_label_array(cls, root_path, chosen_feature_list):
        feature_size = 0
        cycles_feature_list = []
        instructions_feature_list = []
        branch_feature_list = []
        combined_label_list = []

        for chosen_feature in chosen_feature_list:
            if chosen_feature == Constant.LIST_SEQUENCE[0]:  # cycles
                cycles_feature_list = cls.__read_csv(root_path + '/raw_cycles_feature.csv')
                cycles_label_list = cls.__read_csv(root_path + '/raw_cycles_label.csv')
                combined_label_list = cycles_label_list
                feature_size = len(cycles_feature_list)
            elif chosen_feature == Constant.LIST_SEQUENCE[1]:  # instructions
                instructions_feature_list = cls.__read_csv(root_path + '/raw_instructions_feature.csv')
                instructions_label_list = cls.__read_csv(root_path + '/raw_instructions_label.csv')
                combined_label_list = instructions_label_list
                feature_size = len(instructions_feature_list)
            elif chosen_feature == Constant.LIST_SEQUENCE[2]:  # branch
                branch_feature_list = cls.__read_csv(root_path + '/raw_branch_feature.csv')
                branch_label_list = cls.__read_csv(root_path + '/raw_branch_label.csv')
                combined_label_list = branch_label_list
                feature_size = len(branch_feature_list)

        combined_feature_list = []
        for index in range(0, feature_size):
            record_list = []
            for chosen_feature in chosen_feature_list:
                if chosen_feature == Constant.LIST_SEQUENCE[0]:  # cycles
                    temp_list = cycles_feature_list[index]
                    record_list.extend(temp_list)
                elif chosen_feature == Constant.LIST_SEQUENCE[1]:  # instructions
                    temp_list = instructions_feature_list[index]
                    record_list.extend(temp_list)
                elif chosen_feature == Constant.LIST_SEQUENCE[2]:  # branch
                    temp_list = branch_feature_list[index]
                    record_list.extend(temp_list)

            combined_feature_list.append(record_list)

        return combined_feature_list, combined_label_list

    @classmethod
    def divide_training_and_testing_features(cls, feature_list, label_list):
        mixed_list = []
        for index, feature in enumerate(feature_list):
            temp_list = []
            temp_list.extend(feature)
            temp_list.extend(label_list[index])
            mixed_list.append(temp_list)

        np.random.shuffle(mixed_list)

        feature_size = len(mixed_list)
        training_size = round(feature_size * 0.75)
        testing_size = round(feature_size * 0.25)

        training_feature_list = []
        training_label_list = []
        label_index = len(mixed_list[0]) - 1
        for index in range(0, training_size):
            temp_list = [int(data) for data in mixed_list[index][:-1]]
            training_feature_list.append(temp_list)
            training_label_list.append([int(mixed_list[index][label_index])])

        testing_feature_list = []
        testing_label_list = []
        for index in range(training_size, training_size + testing_size):
            temp_list = [int(data) for data in mixed_list[index][:-1]]
            testing_feature_list.append(temp_list)
            testing_label_list.append([int(mixed_list[index][label_index])])

        return np.array(training_feature_list), np.array(training_label_list), np.array(testing_feature_list), \
            np.array(testing_label_list)
