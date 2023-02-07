import csv
import os.path
import statistics as st

import Constant
from DataSave import DataSave
from Machine_learning_algorithms.K_Nearest_Neighbor import KNN


class Consensus(KNN):
    __gs_stat_diff_list = []
    __cs_stat_diff_list = []
    __gs_record_diff_list = []
    __cs_record_diff_list = []
    __gs_record_diff_mean_std = []
    __cs_record_diff_mean_std = []
    __attack_path = None
    __normal_path = None

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

    def __get_chosen_feature_data(self, data_list, chosen_record_diff_list):
        for chosen_symbol in chosen_record_diff_list:
            for record in data_list:
                temp_list = record.split()
                symbol_name = temp_list[0]
                if chosen_symbol == symbol_name:
                    print(temp_list)

    def __get_cs_top_data(self, chosen_record_diff_list):
        normal_cs_path_list = self.__get_file_list_in_dir(self.__normal_path)
        unique_normal_cs_id_list = self.__get_unique_cs_id(normal_cs_path_list)
        normal_cs_top_feature_dict = {}

        for normal_cs_path in normal_cs_path_list:
            if normal_cs_path.find(Constant.CS_TOP) > 0:
                for unique_normal_cs_id in unique_normal_cs_id_list:
                    if normal_cs_path.find(unique_normal_cs_id) > 0:
                        if normal_cs_path.find(Constant.LIST_SEQUENCE[0]) > 0:  # cycles
                            print(unique_normal_cs_id, normal_cs_path)
                            data = self.__read_str_txt(normal_cs_path)
                            self.__get_chosen_feature_data(data, chosen_record_diff_list)
                        elif normal_cs_path.find(Constant.LIST_SEQUENCE[1]) > 0:  # instructions
                            print(unique_normal_cs_id, normal_cs_path)
                            data = self.__read_str_txt(normal_cs_path)
                            self.__get_chosen_feature_data(data, chosen_record_diff_list)
                        elif normal_cs_path.find(Constant.LIST_SEQUENCE[2]) > 0:  # branch
                            print(unique_normal_cs_id, normal_cs_path)
                            data = self.__read_str_txt(normal_cs_path)
                            self.__get_chosen_feature_data(data, chosen_record_diff_list)

    def load_data(self):
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