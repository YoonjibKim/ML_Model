import csv
import os.path
import statistics as st

import numpy as np

import Constant
from Data_Save import DataSave
from Machine_learning_algorithms.K_Nearest_Neighbor import KNN


class Feature_Engineering(KNN):
    __gs_stat_diff_list = []
    __cs_stat_diff_list = []
    __gs_record_diff_list = []
    __cs_record_diff_list = []
    __gs_record_diff_mean_std = []
    __cs_record_diff_mean_std = []
    __attack_path = None
    __normal_path = None
    __cs_normal_array_feature_dict = {}
    __cs_attack_array_feature_dict = {}
    __cs_unique_cs_id_intersection_list = []

    def __init__(self):
        super().__init__()

    @classmethod
    def __get_file_list_in_dir(cls, path):
        filenames = os.listdir(path)
        path_list = []
        for filename in filenames:
            full_filename = os.path.join(path, filename)
            path_list.append(full_filename)

        return path_list

    @classmethod
    def __read_csv(cls, path):
        read_line = []
        with open(path, 'r') as f:
            rdr = csv.reader(f)
            for line in rdr:
                read_line.append(line)

        return read_line

    @classmethod
    def __read_float_txt(cls, path):
        with open(path, 'r') as f:
            read_line = f.readlines()
            read_line = list(map(lambda s: float(s.strip().replace(',', '')), read_line))

        return read_line

    @classmethod
    def __read_str_txt(cls, path):
        with open(path, 'r') as f:
            read_line_list = f.readlines()

        temp_list = []
        for read_line in read_line_list:
            temp_list.append(read_line.strip())

        return temp_list

    @classmethod
    def __get_gs_record_overhead_diff(cls, normal_list, attack_list):
        percent_list = []
        for normal_item in normal_list:
            normal_symbol_name = normal_item[1]
            for attack_item in attack_list:
                attack_symbol_name = attack_item[1]
                if normal_symbol_name == attack_symbol_name:
                    normal_percent_str = normal_item[0][:-1]
                    attack_percent_str = attack_item[0][:-1]
                    normal_percent = float(normal_percent_str)
                    attack_percent = float(attack_percent_str)
                    percent_diff = abs(normal_percent - attack_percent)
                    round_percent_diff = round(percent_diff, 3)
                    percent_list.append([round_percent_diff, normal_symbol_name])

        percent_list = sorted(percent_list, key=lambda _percent_list: _percent_list[0], reverse=True)
        return percent_list

    @classmethod
    def __get_overhead_mean(cls, data_list):
        temp_list = []
        for data in data_list:
            temp_list.append(data[0])

        arith_mean = st.mean(temp_list)
        return arith_mean, st.stdev(temp_list)

    def __get_sorted_gs_record_overhead_diff(self, normal_path_list, attack_path_list):
        normal_cycles_list = []
        attack_cycles_list = []
        normal_instructions_list = []
        attack_instructions_list = []
        normal_branch_list = []
        attack_branch_list = []

        for path in normal_path_list:  # normal
            if path.find(Constant.GS_RECORD) > 0:
                if path.find(Constant.LIST_SEQUENCE[0]) > 0:  # cycles
                    normal_cycles_list = self.__read_csv(path)
                elif path.find(Constant.LIST_SEQUENCE[1]) > 0:  # instructions
                    normal_instructions_list = self.__read_csv(path)
                elif path.find(Constant.LIST_SEQUENCE[2]) > 0:  # branch
                    normal_branch_list = self.__read_csv(path)

        for path in attack_path_list:  # attack
            if path.find(Constant.GS_RECORD) > 0:
                if path.find(Constant.LIST_SEQUENCE[0]) > 0:  # cycles
                    attack_cycles_list = self.__read_csv(path)
                elif path.find(Constant.LIST_SEQUENCE[1]) > 0:  # instructions
                    attack_instructions_list = self.__read_csv(path)
                elif path.find(Constant.LIST_SEQUENCE[2]) > 0:  # branch
                    attack_branch_list = self.__read_csv(path)

        cycles_list = self.__get_gs_record_overhead_diff(normal_cycles_list, attack_cycles_list)
        instructions_list = self.__get_gs_record_overhead_diff(normal_instructions_list, attack_instructions_list)
        branch_list = self.__get_gs_record_overhead_diff(normal_branch_list, attack_branch_list)

        sorted_cycles_list = sorted(cycles_list, key=lambda _cycles_list: _cycles_list[0], reverse=True)
        sorted_instructions_list = sorted(instructions_list, key=lambda _instructions_list: _instructions_list[0],
                                          reverse=True)
        sorted_branch_list = sorted(branch_list, key=lambda _branch_list: _branch_list[0], reverse=True)

        return sorted_cycles_list, sorted_instructions_list, sorted_branch_list

    @classmethod
    def __get_cs_record_intersection(cls, data_list):
        intersection_set = {}
        for inner_normal_symbol_list in data_list:
            symbol_list = []
            for normal_symbol in inner_normal_symbol_list:
                symbol_list.append(normal_symbol[1])
            if len(intersection_set) == 0:
                intersection_set = set(symbol_list)
            else:
                intersection_set &= set(symbol_list)

        return list(intersection_set)

    @classmethod
    def __get_cs_record_base_data(cls, param_list, base_list):
        data_list = []
        for base in base_list:
            temp_list = []
            for cs_normal_list in param_list:
                for normal_data in cs_normal_list:
                    symbol_name = normal_data[1]
                    if symbol_name == base:
                        temp_list.append(float(normal_data[0].strip('%')))
            data_list.append(st.mean(temp_list))

        return data_list

    def __get_cs_record_overhead_diff(self, normal_list, attack_list, base_list):
        normal_data_base_list = self.__get_cs_record_base_data(normal_list, base_list)
        attack_data_base_list = self.__get_cs_record_base_data(attack_list, base_list)

        diff_list = []
        for normal_data, attack_data, base in zip(normal_data_base_list, attack_data_base_list, base_list):
            diff = normal_data - attack_data
            diff_list.append([round(abs(diff), 3), base])

        return diff_list

    def __get_sorted_cs_record_overhead_diff(self, normal_path_list, attack_path_list):
        normal_cycles_list = []
        attack_cycles_list = []
        normal_instructions_list = []
        attack_instructions_list = []
        normal_branch_list = []
        attack_branch_list = []

        for normal_path in normal_path_list:
            if normal_path.find(Constant.CS_RECORD) > 0:
                if normal_path.find(Constant.LIST_SEQUENCE[0]) > 0:  # cycles
                    normal_cycles = self.__read_csv(normal_path)
                    normal_cycles_list.append(normal_cycles)
                elif normal_path.find(Constant.LIST_SEQUENCE[1]) > 0:  # instructions
                    normal_instructions = self.__read_csv(normal_path)
                    normal_instructions_list.append(normal_instructions)
                elif normal_path.find(Constant.LIST_SEQUENCE[2]) > 0:  # branch
                    normal_branch = self.__read_csv(normal_path)
                    normal_branch_list.append(normal_branch)

        for attack_path in attack_path_list:
            if attack_path.find(Constant.CS_RECORD) > 0:
                if attack_path.find(Constant.LIST_SEQUENCE[0]) > 0:  # cycles
                    attack_cycles = self.__read_csv(attack_path)
                    attack_cycles_list.append(attack_cycles)
                elif attack_path.find(Constant.LIST_SEQUENCE[1]) > 0:  # instructions
                    attack_instructions = self.__read_csv(attack_path)
                    attack_instructions_list.append(attack_instructions)
                elif attack_path.find(Constant.LIST_SEQUENCE[2]) > 0:  # branch
                    attack_branch = self.__read_csv(attack_path)
                    attack_branch_list.append(attack_branch)

        normal_cycle_intersection_list = self.__get_cs_record_intersection(normal_cycles_list)
        attack_cycle_intersection_list = self.__get_cs_record_intersection(attack_cycles_list)
        both_cycles_intersection_list = set(normal_cycle_intersection_list) & set(attack_cycle_intersection_list)

        normal_instructions_intersection_list = self.__get_cs_record_intersection(normal_instructions_list)
        attack_instructions_intersection_list = self.__get_cs_record_intersection(attack_instructions_list)
        both_instructions_intersection_list = set(normal_instructions_intersection_list) & \
                                              set(attack_instructions_intersection_list)

        normal_branch_intersection_list = self.__get_cs_record_intersection(normal_branch_list)
        attack_branch_intersection_list = self.__get_cs_record_intersection(attack_branch_list)
        both_branch_intersection_list = set(normal_branch_intersection_list) & \
                                        set(attack_branch_intersection_list)

        diff_cycles_list = self.__get_cs_record_overhead_diff(normal_cycles_list, attack_cycles_list,
                                                              both_cycles_intersection_list)
        diff_instructions_list = self.__get_cs_record_overhead_diff(normal_instructions_list, attack_instructions_list,
                                                                    both_instructions_intersection_list)
        diff_branch_list = self.__get_cs_record_overhead_diff(normal_branch_list, attack_branch_list,
                                                              both_branch_intersection_list)

        sorted_diff_cycles_list = sorted(diff_cycles_list, key=lambda _diff_cycles_list: _diff_cycles_list[0],
                                         reverse=True)
        sorted_diff_instructions_list = \
            sorted(diff_instructions_list, key=lambda _diff_instructions_list: _diff_instructions_list[0], reverse=True)
        sorted_diff_branch_list = sorted(diff_branch_list, key=lambda _diff_branch_list: _diff_branch_list[0],
                                         reverse=True)

        return sorted_diff_cycles_list, sorted_diff_instructions_list, sorted_diff_branch_list

    def __get_gs_stat_geo_mean(self, path_list):
        cycles_geo_mean = 0
        instructions_geo_mean = 0
        branch_geo_mean = 0

        for path in path_list:
            if path.find(Constant.GS_STAT) > 0:
                if path.find(Constant.LIST_SEQUENCE[0]) > 0:  # cycles
                    cycles_list = self.__read_float_txt(path)
                    cycles_geo_mean = st.geometric_mean(cycles_list)
                elif path.find(Constant.LIST_SEQUENCE[1]) > 0:  # instructions
                    instructions_list = self.__read_float_txt(path)
                    instructions_geo_mean = st.geometric_mean(instructions_list)
                elif path.find(Constant.LIST_SEQUENCE[2]) > 0:  # branch
                    branch_list = self.__read_float_txt(path)
                    branch_geo_mean = st.geometric_mean(branch_list)

        return cycles_geo_mean, instructions_geo_mean, branch_geo_mean

    def __get_cs_stat_geo_mean(self, path_list):
        cycles_list = []
        instructions_list = []
        branch_list = []

        for path in path_list:
            if path.find(Constant.CS_STAT) > 0:
                if path.find(Constant.LIST_SEQUENCE[0]) > 0:  # cycles
                    data_list = self.__read_float_txt(path)
                    cycles_list.extend(data_list)
                elif path.find(Constant.LIST_SEQUENCE[1]) > 0:  # instructions
                    data_list = self.__read_float_txt(path)
                    instructions_list.extend(data_list)
                elif path.find(Constant.LIST_SEQUENCE[2]) > 0:  # branch
                    data_list = self.__read_float_txt(path)
                    branch_list.extend(data_list)

        cycles_geo_mean = st.geometric_mean(cycles_list)
        instruction_geo_mean = st.geometric_mean(instructions_list)
        branch_geo_mean = st.geometric_mean(branch_list)

        return cycles_geo_mean, instruction_geo_mean, branch_geo_mean

    @classmethod
    def __get_mean_std(cls, data_list):
        temp_list = []
        for data in data_list:
            temp_list.append(data[0])

        return round(st.mean(temp_list), 3), round(st.stdev(temp_list), 3)

    def parsing_dataset(self, root_path):
        self.__normal_path = root_path + '/' + Constant.NORMAL
        self.__attack_path = root_path + '/' + Constant.ATTACK
        normal_path_list = self.__get_file_list_in_dir(self.__normal_path)
        attack_path_list = self.__get_file_list_in_dir(self.__attack_path)

        gs_normal_cycles_geo_mean, gs_normal_instructions_geo_mean, gs_normal_branch_geo_mean = \
            self.__get_gs_stat_geo_mean(normal_path_list)
        gs_attack_cycles_geo_mean, gs_attack_instructions_geo_mean, gs_attack_branch_geo_mean = \
            self.__get_gs_stat_geo_mean(attack_path_list)

        gs_stat_cycle_delta = abs(gs_normal_cycles_geo_mean - gs_attack_cycles_geo_mean)
        gs_stat_instruction_delta = abs(gs_normal_instructions_geo_mean - gs_attack_instructions_geo_mean)
        gs_stat_branch_delta = abs(gs_normal_branch_geo_mean - gs_attack_branch_geo_mean)

        self.__gs_stat_diff_list = [[round(gs_stat_cycle_delta, 3), Constant.LIST_SEQUENCE[0]],
                                    [round(gs_stat_instruction_delta, 3), Constant.LIST_SEQUENCE[1]],
                                    [round(gs_stat_branch_delta, 3), Constant.LIST_SEQUENCE[2]]]
        self.__gs_stat_diff_list = sorted(self.__gs_stat_diff_list, key=lambda gs_stat_diff_list: gs_stat_diff_list[0],
                                          reverse=True)

        cs_normal_cycles_geo_mean, cs_normal_instruction_geo_mean, cs_normal_branch_geo_mean = \
            self.__get_cs_stat_geo_mean(normal_path_list)
        cs_attack_cycles_geo_mean, cs_attack_instruction_geo_mean, cs_attack_branch_geo_mean = \
            self.__get_cs_stat_geo_mean(attack_path_list)

        cs_stat_cycle_delta = abs(cs_normal_cycles_geo_mean - cs_attack_cycles_geo_mean)
        cs_stat_instruction_delta = abs(cs_normal_instruction_geo_mean - cs_attack_instruction_geo_mean)
        cs_stat_branch_delta = abs(cs_normal_branch_geo_mean - cs_attack_branch_geo_mean)

        self.__cs_stat_diff_list = [[round(cs_stat_cycle_delta, 3), Constant.LIST_SEQUENCE[0]],
                                    [round(cs_stat_instruction_delta, 3), Constant.LIST_SEQUENCE[1]],
                                    [round(cs_stat_branch_delta, 3), Constant.LIST_SEQUENCE[2]]]
        self.__cs_stat_diff_list = sorted(self.__cs_stat_diff_list, key=lambda cs_stat_diff_list: cs_stat_diff_list[0],
                                          reverse=True)

        sorted_diff_gs_cycles_list, sorted_diff_gs_instructions_list, sorted_diff_gs_branch_list = \
            self.__get_sorted_gs_record_overhead_diff(normal_path_list, attack_path_list)
        sorted_diff_cs_cycles_list, sorted_diff_cs_instructions_list, sorted_diff_cs_branch_list = \
            self.__get_sorted_cs_record_overhead_diff(normal_path_list, attack_path_list)

        self.__gs_record_diff_list = [sorted_diff_gs_cycles_list, sorted_diff_gs_instructions_list,
                                      sorted_diff_gs_branch_list]
        self.__cs_record_diff_list = [sorted_diff_cs_cycles_list, sorted_diff_cs_instructions_list,
                                      sorted_diff_cs_branch_list]

        self.__gs_record_diff_mean_std = [[self.__get_mean_std(sorted_diff_gs_cycles_list), Constant.LIST_SEQUENCE[0]],
                                          [self.__get_mean_std(sorted_diff_gs_instructions_list),
                                           Constant.LIST_SEQUENCE[1]],
                                          [self.__get_mean_std(sorted_diff_gs_branch_list), Constant.LIST_SEQUENCE[2]]]

        self.__cs_record_diff_mean_std = [[self.__get_mean_std(sorted_diff_gs_cycles_list), Constant.LIST_SEQUENCE[0]],
                                          [self.__get_mean_std(sorted_diff_gs_instructions_list),
                                           Constant.LIST_SEQUENCE[1]],
                                          [self.__get_mean_std(sorted_diff_gs_branch_list), Constant.LIST_SEQUENCE[2]]]

        DataSave.save_profiling_data(self.__gs_stat_diff_list, self.__cs_stat_diff_list,
                                     self.__gs_record_diff_list, self.__cs_record_diff_list,
                                     self.__gs_record_diff_mean_std, self.__cs_record_diff_mean_std)

    @classmethod
    def __get_unique_cs_id(cls, path_list):
        cs_id_list = []
        for path in path_list:
            if path.find(Constant.CS_TOP) > 0:
                temp_path_list = path.split('_')
                cs_id_list.append(temp_path_list[len(temp_path_list) - 1][:-4])

        return list(set(cs_id_list))

    @classmethod
    def __get_head_symbol(cls, data_list):
        temp_list = []
        for data in data_list:
            split_list = data.split()
            temp_list.append(split_list[0])

        return temp_list

    def __get_unique_cs_top_symbol(self, file_path):
        file_list = self.__get_file_list_in_dir(file_path)
        cycles_list = []
        instructions_list = []
        branch_list = []
        for file in file_list:
            if file.find(Constant.CS_TOP) > 0:
                if file.find(Constant.LIST_SEQUENCE[0]) > 0:
                    data = self.__read_str_txt(file)
                    symbol = self.__get_head_symbol(data)
                    cycles_list.extend(symbol)
                elif file.find(Constant.LIST_SEQUENCE[1]) > 0:
                    data = self.__read_str_txt(file)
                    symbol = self.__get_head_symbol(data)
                    instructions_list.extend(symbol)
                elif file.find(Constant.LIST_SEQUENCE[2]) > 0:
                    data = self.__read_str_txt(file)
                    symbol = self.__get_head_symbol(data)
                    branch_list.extend(symbol)

        return list(set(cycles_list)), list(set(instructions_list)), list(set(branch_list))

    @classmethod
    def __get_cs_top_symbol_intersection(cls, normal_data_list, attack_data_list):
        normal_data_set = set(normal_data_list)
        attack_data_set = set(attack_data_list)
        intersection_set = normal_data_set & attack_data_set

        return list(intersection_set)

    @classmethod
    def __get_chosen_feature_data(cls, data_list, chosen_record_diff_list):
        ret_data_list = []
        for chosen_symbol in chosen_record_diff_list:
            for record in data_list:
                temp_list = record.split()
                symbol_name = temp_list[0]
                if chosen_symbol == symbol_name:
                    ret_data_list.append(temp_list)

        return ret_data_list

    def __get_cs_top_unique_cs_id_data(self, unique_cs_id_list, cs_path_list, chosen_record_diff_list):
        sampled_cycles_dict = {}
        sampled_instructions_dict = {}
        sampled_branch_dict = {}
        for cs_path in cs_path_list:
            if cs_path.find(Constant.CS_TOP) > 0:
                for unique_cs_id in unique_cs_id_list:
                    if cs_path.find(unique_cs_id) > 0:
                        if cs_path.find(Constant.LIST_SEQUENCE[0]) > 0:  # cycles
                            data = self.__read_str_txt(cs_path)
                            data_list = self.__get_chosen_feature_data(data, chosen_record_diff_list)
                            sampled_cycles_dict[unique_cs_id] = data_list
                        elif cs_path.find(Constant.LIST_SEQUENCE[1]) > 0:  # instructions
                            data = self.__read_str_txt(cs_path)
                            data_list = self.__get_chosen_feature_data(data, chosen_record_diff_list)
                            sampled_instructions_dict[unique_cs_id] = data_list
                        elif cs_path.find(Constant.LIST_SEQUENCE[2]) > 0:  # branch
                            data = self.__read_str_txt(cs_path)
                            data_list = self.__get_chosen_feature_data(data, chosen_record_diff_list)
                            sampled_branch_dict[unique_cs_id] = data_list

        return sampled_cycles_dict, sampled_instructions_dict, sampled_branch_dict

    @classmethod
    def __get_cs_top_intersection_symbol_list(cls, normal_ref_feature_dict, attack_ref_feature_dict,
                                              base_intersection_list):
        normal_cs_id_list = []
        for cs_id in normal_ref_feature_dict.keys():
            normal_cs_id_list.append(cs_id)
        attack_cs_id_list = []
        for cs_id in attack_ref_feature_dict.keys():
            attack_cs_id_list.append(cs_id)

        cs_id_intersection_set = set(normal_cs_id_list) & set(attack_cs_id_list)
        unique_cs_id_intersection_list = list(cs_id_intersection_set)
        cls.__cs_unique_cs_id_intersection_list = unique_cs_id_intersection_list

        intersection_symbol_set = {}
        for unique_cs_id in unique_cs_id_intersection_list:
            normal_ref_feature_list = normal_ref_feature_dict[unique_cs_id]
            attack_ref_feature_list = attack_ref_feature_dict[unique_cs_id]

            normal_symbol_list = []
            for normal_ref_feature in normal_ref_feature_list:
                normal_symbol = normal_ref_feature[0]
                for base in base_intersection_list:
                    if base == normal_symbol:
                        normal_symbol_list.append(normal_symbol)
                        break

            attack_symbol_list = []
            for attack_ref_feature in attack_ref_feature_list:
                attack_symbol = attack_ref_feature[0]
                for base in base_intersection_list:
                    if base == attack_symbol:
                        attack_symbol_list.append(attack_symbol)
                        break

            sub_intersection_symbol_set = set(normal_symbol_list) & set(attack_symbol_list)
            if len(intersection_symbol_set) > 0:
                intersection_symbol_set &= sub_intersection_symbol_set
            else:
                intersection_symbol_set = sub_intersection_symbol_set

        return list(intersection_symbol_set)

    @classmethod
    def __convert_string_to_nparray(cls, base_sorted_feature_diff_list, normal_feature_dict, attack_feature_dict):
        normal_array_feature_dict = {}
        for base in base_sorted_feature_diff_list:
            symbol = base[0]
            if symbol in normal_feature_dict:
                normal_datapoint_list = normal_feature_dict[symbol]
                normal_array_datapoint_array = np.float_(normal_datapoint_list)
                normal_array_feature_dict[symbol] = normal_array_datapoint_array

        attack_array_feature_dict = {}
        for base in base_sorted_feature_diff_list:
            symbol = base[0]
            if symbol in attack_feature_dict:
                attack_datapoint_list = attack_feature_dict[symbol]
                attack_array_datapoint_array = np.float_(attack_datapoint_list)
                attack_array_feature_dict[symbol] = attack_array_datapoint_array

        return normal_array_feature_dict, attack_array_feature_dict

    def __get_cs_top_data(self, chosen_record_diff_list):
        normal_cycles_list, normal_instructions_list, normal_branch_list = \
            self.__get_unique_cs_top_symbol(self.__normal_path)
        attack_cycles_list, attack_instructions_list, attack_branch_list = \
            self.__get_unique_cs_top_symbol(self.__attack_path)

        cycles_base_intersection_list = self.__get_cs_top_symbol_intersection(normal_cycles_list, attack_cycles_list)
        instructions_base_intersection_list = self.__get_cs_top_symbol_intersection(normal_instructions_list,
                                                                                    attack_instructions_list)
        branch_base_intersection_list = self.__get_cs_top_symbol_intersection(normal_branch_list, attack_branch_list)

        normal_cs_path_list = self.__get_file_list_in_dir(self.__normal_path)
        unique_normal_cs_id_list = self.__get_unique_cs_id(normal_cs_path_list)
        attack_cs_path_list = self.__get_file_list_in_dir(self.__attack_path)
        unique_attack_cs_id_list = self.__get_unique_cs_id(attack_cs_path_list)

        normal_ref_cycles_dict, normal_ref_instructions_dict, normal_ref_branch_dict = \
            self.__get_cs_top_unique_cs_id_data(unique_normal_cs_id_list, normal_cs_path_list, chosen_record_diff_list)
        attack_ref_cycles_dict, attack_ref_instructions_dict, attack_ref_branch_dict = \
            self.__get_cs_top_unique_cs_id_data(unique_attack_cs_id_list, attack_cs_path_list, chosen_record_diff_list)

        intersection_cycles_list = self.__get_cs_top_intersection_symbol_list(normal_ref_cycles_dict,
                                                                              attack_ref_cycles_dict,
                                                                              cycles_base_intersection_list)
        intersection_instructions_list = self.__get_cs_top_intersection_symbol_list(normal_ref_instructions_dict,
                                                                                    attack_ref_instructions_dict,
                                                                                    instructions_base_intersection_list)
        intersection_branch_list = self.__get_cs_top_intersection_symbol_list(normal_ref_branch_dict,
                                                                              attack_ref_branch_dict,
                                                                              branch_base_intersection_list)

        normal_cycle_feature_dict, attack_cycle_feature_dict = self.__get_ml_features(normal_ref_cycles_dict,
                                                                                      attack_ref_cycles_dict,
                                                                                      intersection_cycles_list)
        normal_instructions_feature_dict, attack_instructions_feature_dict = \
            self.__get_ml_features(normal_ref_instructions_dict, attack_ref_instructions_dict,
                                   intersection_instructions_list)
        normal_branch_feature_dict, attack_branch_feature_dict = \
            self.__get_ml_features(normal_ref_branch_dict, attack_ref_branch_dict, intersection_branch_list)

        sorted_cycles_diff_list, diff_total_cycles_mean, diff_total_cycles_std = \
            self.__get_feature_difference(normal_cycle_feature_dict, attack_cycle_feature_dict)
        sorted_instructions_diff_list, diff_total_instructions_mean, diff_total_instructions_std = \
            self.__get_feature_difference(normal_instructions_feature_dict, attack_instructions_feature_dict)
        sorted_branch_diff_list, diff_total_branch_mean, diff_total_branch_std = \
            self.__get_feature_difference(normal_branch_feature_dict, attack_branch_feature_dict)

        # data save, mean, std, difference

        if self.__cs_stat_diff_list[0][1] == Constant.LIST_SEQUENCE[0]:
            self.__cs_normal_array_feature_dict, self.__cs_attack_array_feature_dict = \
                self.__convert_string_to_nparray(sorted_cycles_diff_list, normal_cycle_feature_dict,
                                                 attack_cycle_feature_dict)
        elif self.__cs_stat_diff_list[0][1] == Constant.LIST_SEQUENCE[1]:
            self.__cs_normal_array_feature_dict, self.__cs_attack_array_feature_dict = \
                self.__convert_string_to_nparray(sorted_instructions_diff_list, normal_instructions_feature_dict,
                                                 attack_instructions_feature_dict)
        elif self.__cs_stat_diff_list[0][1] == Constant.LIST_SEQUENCE[2]:
            self.__cs_normal_array_feature_dict, self.__cs_attack_array_feature_dict = \
                self.__convert_string_to_nparray(sorted_branch_diff_list, normal_branch_feature_dict,
                                                 attack_branch_feature_dict)

    @classmethod
    def __get_feature_difference(cls, normal_feature_dict, attack_feature_dict):
        sub_diff_list = []
        total_diff_mean_list = []
        for normal_symbol, normal_values in normal_feature_dict.items():
            attack_values = attack_feature_dict[normal_symbol]
            attack_float_values = np.float_(attack_values)
            attack_mean = np.mean(attack_float_values)
            normal_float_values = np.float_(normal_values)
            normal_mean = np.mean(normal_float_values)
            diff_mean = abs(normal_mean - attack_mean)
            sub_diff_list.append([normal_symbol, diff_mean])
            total_diff_mean_list.append(diff_mean)

        sorted_sub_diff_list = sorted(sub_diff_list, key=lambda _sub_diff_list: _sub_diff_list[1], reverse=True)
        return sorted_sub_diff_list, np.mean(total_diff_mean_list), np.std(total_diff_mean_list)

    @classmethod
    def __get_ml_features(cls, normal_ref_feature_dict, attack_ref_feature_dict, intersection_feature_list):
        normal_feature_dict = {}
        for normal_cs_id, normal_ref_feature in normal_ref_feature_dict.items():
            for intersection_cs_id in cls.__cs_unique_cs_id_intersection_list:
                if normal_cs_id == intersection_cs_id:
                    for normal_feature in normal_ref_feature:
                        symbol = normal_feature[0]
                        for intersection_feature in intersection_feature_list:
                            if intersection_feature == symbol:
                                normal_feature_dict[symbol] = normal_feature[1:]

        attack_feature_dict = {}
        for attack_cs_id, attack_ref_feature in attack_ref_feature_dict.items():
            for intersection_cs_id in cls.__cs_unique_cs_id_intersection_list:
                if attack_cs_id == intersection_cs_id:
                    for attack_feature in attack_ref_feature:
                        symbol = attack_feature[0]
                        for intersection_feature in intersection_feature_list:
                            if intersection_feature == symbol:
                                attack_feature_dict[symbol] = attack_feature[1:]

        return normal_feature_dict, attack_feature_dict

    def load_data_for_cs(self):
        largest_stat_feature_list = self.__cs_stat_diff_list[0][1]
        temp_record_diff_list = []
        chosen_record_diff_list = []
        chosen_mean = 0

        if largest_stat_feature_list == Constant.LIST_SEQUENCE[0]:  # cycles
            temp_record_diff_list = self.__cs_record_diff_list[0]
            chosen_mean = self.__cs_record_diff_mean_std[0][0][0]
        elif largest_stat_feature_list == Constant.LIST_SEQUENCE[1]:  # instructions
            temp_record_diff_list = self.__cs_record_diff_list[1]
            chosen_mean = self.__cs_record_diff_mean_std[1][0][0]
        elif largest_stat_feature_list == Constant.LIST_SEQUENCE[2]:  # branch
            temp_record_diff_list = self.__cs_record_diff_list[2]
            chosen_mean = self.__cs_record_diff_mean_std[2][0][0]

        for temp_record_diff in temp_record_diff_list:
            mean_value = temp_record_diff[0]
            # if mean_value > chosen_mean:
            chosen_record_diff_list.append(temp_record_diff[1])

        self.__get_cs_top_data(chosen_record_diff_list)

    def __get_testing_data(self, cs_id_list, root_path):
        feature_string_list = []
        for file_path in self.__get_file_list_in_dir(root_path):
            if file_path.find(Constant.CS_TOP) > 0:
                temp_list = file_path.split(Constant.CS_TOP)
                temp = temp_list[len(temp_list) - 1]
                temp_list = temp.split('_')
                temp = temp_list[len(temp_list) - 1]
                temp_list = temp.split('.')
                temp_cs_id = temp_list[0]
                if temp_cs_id not in self.__cs_unique_cs_id_intersection_list:
                    choice = self.__cs_stat_diff_list[0][1]
                    if file_path.find(choice) > 0:
                        feature_string_list = self.__read_str_txt(file_path)

        feature_array_dict = {}
        for cs_id in cs_id_list:
            for feature_string in feature_string_list:
                temp_list = feature_string.split()
                symbol = temp_list[0]
                if cs_id == symbol:
                    feature_array_dict[symbol] = np.float_(temp_list[1:])

        return feature_array_dict

    def get_feature_for_cs(self):
        temp_training_normal_data_dict = self.__cs_normal_array_feature_dict
        temp_training_attack_data_dict = self.__cs_attack_array_feature_dict

        symbol_list = list(cs_id for cs_id in temp_training_normal_data_dict.keys())
        temp_testing_normal_data_dict = self.__get_testing_data(symbol_list, self.__normal_path)
        testing_normal_symbol_list = list(symbol for symbol in temp_testing_normal_data_dict.keys())
        temp_testing_attack_data_dict = self.__get_testing_data(symbol_list, self.__attack_path)
        testing_attack_symbol_list = list(symbol for symbol in temp_testing_attack_data_dict.keys())
        testing_intersection_symbol_list = list(set(testing_normal_symbol_list) & set(testing_attack_symbol_list))

        training_normal_symbol_list = list(symbol for symbol in temp_training_normal_data_dict.keys())
        training_attack_symbol_list = list(symbol for symbol in temp_training_attack_data_dict.keys())
        training_intersection_symbol_list = list(set(training_normal_symbol_list) & set(training_attack_symbol_list))
        intersection_symbol_list = list(set(testing_intersection_symbol_list) & set(training_intersection_symbol_list))

        training_normal_data_dict = {}
        training_attack_data_dict = {}
        testing_normal_data_dict = {}
        testing_attack_data_dict = {}
        for symbol in intersection_symbol_list:
            training_normal_data_dict[symbol] = temp_training_normal_data_dict[symbol]
            training_attack_data_dict[symbol] = temp_training_attack_data_dict[symbol]
            testing_normal_data_dict[symbol] = temp_testing_normal_data_dict[symbol]
            testing_attack_data_dict[symbol] = temp_testing_attack_data_dict[symbol]

        return training_normal_data_dict, training_attack_data_dict, testing_normal_data_dict, \
            testing_attack_data_dict, intersection_symbol_list

    @classmethod
    def __get_larger_avg_features(cls, data_dict):
        temp_size_list = []
        for data_points in data_dict.values():
            temp_size_list.append(len(data_points))
        avg = np.mean(temp_size_list)
        temp_size_list.clear()

        symbol_list = []
        for symbol, data_points in data_dict.items():
            data_size = len(data_points)
            if data_size > avg:
                symbol_list.append(symbol)

        return symbol_list

    @classmethod
    def __get_min_size_feature(cls, data_dict, all_intersection_symbol_list):
        temp_size_list = []
        for symbol in all_intersection_symbol_list:
            temp_size = len(data_dict[symbol])
            temp_size_list.append(temp_size)
        temp_size_list = sorted(temp_size_list)

        return temp_size_list[0]

    def get_cut_feature_for_cs(self, training_normal_data_dict, training_attack_data_dict, testing_normal_data_dict,
                               testing_attack_data_dict, intersection_symbol_list):
        training_normal_symbol_list = self.__get_larger_avg_features(training_normal_data_dict)
        training_attack_symbol_list = self.__get_larger_avg_features(training_attack_data_dict)
        testing_normal_symbol_list = self.__get_larger_avg_features(testing_normal_data_dict)
        testing_attack_symbol_list = self.__get_larger_avg_features(testing_attack_data_dict)

        all_intersection_symbol_set = set(training_normal_symbol_list) & set(training_attack_symbol_list) & \
                                      set(testing_normal_symbol_list) & set(testing_attack_symbol_list) & \
                                      set(intersection_symbol_list)
        all_intersection_symbol_list = list(all_intersection_symbol_set)

        training_normal_size = self.__get_min_size_feature(training_normal_data_dict, all_intersection_symbol_list)
        training_attack_size = self.__get_min_size_feature(training_attack_data_dict, all_intersection_symbol_list)
        testing_normal_size = self.__get_min_size_feature(testing_normal_data_dict, all_intersection_symbol_list)
        testing_attack_size = self.__get_min_size_feature(testing_attack_data_dict, all_intersection_symbol_list)

        if training_normal_size > training_attack_size:
            training_min_size = training_attack_size
        else:
            training_min_size = training_normal_size

        if testing_normal_size > testing_attack_size:
            testing_min_size = testing_attack_size
        else:
            testing_min_size = testing_normal_size

        training_normal_cut_data_dict = {}
        training_attack_cut_data_dict = {}
        testing_normal_cut_data_dict = {}
        testing_attack_cut_data_dict = {}
        for symbol in all_intersection_symbol_list:
            data_list = training_normal_data_dict[symbol][:training_min_size]
            training_normal_cut_data_dict[symbol] = data_list

            data_list = training_attack_data_dict[symbol][:training_min_size]
            training_attack_cut_data_dict[symbol] = data_list

            data_list = testing_normal_data_dict[symbol][:testing_min_size]
            testing_normal_cut_data_dict[symbol] = data_list

            data_list = testing_attack_data_dict[symbol][:testing_min_size]
            testing_attack_cut_data_dict[symbol] = data_list

        return training_normal_cut_data_dict, training_attack_cut_data_dict, testing_normal_cut_data_dict, \
            testing_attack_cut_data_dict
