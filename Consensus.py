import csv
import os.path
import statistics as st

import Constant
from Machine_learning_algorithms.K_Nearest_Neighbor import KNN


class Consensus(KNN):
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
    def __get_overhead_diff(cls, normal_list, attack_list):
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

    def __get_large_gs_overhead(self, normal_path_list, attack_path_list):
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

        cycles_list = self.__get_overhead_diff(normal_cycles_list, attack_cycles_list)
        instructions_list = self.__get_overhead_diff(normal_instructions_list, attack_instructions_list)
        branch_list = self.__get_overhead_diff(normal_branch_list, attack_branch_list)

        std_list = []
        mean, std = self.__get_overhead_mean(cycles_list)
        std_list.append([mean - std, Constant.LIST_SEQUENCE[0]])
        mean, std = self.__get_overhead_mean(instructions_list)
        std_list.append([mean - std, Constant.LIST_SEQUENCE[1]])
        mean, std = self.__get_overhead_mean(branch_list)
        std_list.append([mean - std, Constant.LIST_SEQUENCE[2]])

        sorted_std_list = sorted(std_list, key=lambda _std_list: _std_list[1], reverse=True)
        return sorted_std_list[0][1]

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

    def __get_large_cs_overhead(self, normal_path_list, attack_path_list):
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

        return None

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

    def load_dataset(self, root_path):
        normal_path = root_path + '/' + Constant.NORMAL
        attack_path = root_path + '/' + Constant.ATTACK
        normal_path_list = self.__get_file_list_in_dir(normal_path)
        attack_path_list = self.__get_file_list_in_dir(attack_path)

        gs_normal_cycles_geo_mean, gs_normal_instructions_geo_mean, gs_normal_branch_geo_mean = \
            self.__get_gs_stat_geo_mean(normal_path_list)
        gs_attack_cycles_geo_mean, gs_attack_instructions_geo_mean, gs_attack_branch_geo_mean = \
            self.__get_gs_stat_geo_mean(attack_path_list)

        gs_stat_cycle_delta = abs(gs_normal_cycles_geo_mean - gs_attack_cycles_geo_mean)
        gs_stat_instruction_delta = abs(gs_normal_instructions_geo_mean - gs_attack_instructions_geo_mean)
        gs_stat_branch_delta = abs(gs_normal_branch_geo_mean - gs_attack_branch_geo_mean)

        gs_stat_diff_list = [[gs_stat_cycle_delta, Constant.LIST_SEQUENCE[0]],
                             [gs_stat_instruction_delta, Constant.LIST_SEQUENCE[1]],
                             [gs_stat_branch_delta, Constant.LIST_SEQUENCE[2]]]

        gs_stat_diff_list = sorted(gs_stat_diff_list, key=lambda _stat_diff_list: _stat_diff_list[0], reverse=True)
        gs_largest_consumed_resource = gs_stat_diff_list[0][1]

        cs_normal_cycles_geo_mean, cs_normal_instruction_geo_mean, cs_normal_branch_geo_mean = \
            self.__get_cs_stat_geo_mean(normal_path_list)
        cs_attack_cycles_geo_mean, cs_attack_instruction_geo_mean, cs_attack_branch_geo_mean = \
            self.__get_cs_stat_geo_mean(attack_path_list)

        cs_stat_cycle_delta = abs(cs_normal_cycles_geo_mean - cs_attack_cycles_geo_mean)
        cs_stat_instruction_delta = abs(cs_normal_instruction_geo_mean - cs_attack_instruction_geo_mean)
        cs_stat_branch_delta = abs(cs_normal_branch_geo_mean - cs_attack_branch_geo_mean)
        cs_stat_diff_list = [[cs_stat_cycle_delta, Constant.LIST_SEQUENCE[0]],
                             [cs_stat_instruction_delta, Constant.LIST_SEQUENCE[1]],
                             [cs_stat_branch_delta, Constant.LIST_SEQUENCE[2]]]

        cs_stat_diff_list = sorted(cs_stat_diff_list, key=lambda _stat_diff_list: _stat_diff_list[0], reverse=True)
        cs_largest_consumed_resource = cs_stat_diff_list[0][1]

        chosen_gs_type = self.__get_large_gs_overhead(normal_path_list, attack_path_list)
        chosen_cs_type = self.__get_large_cs_overhead(normal_path_list, attack_path_list)
