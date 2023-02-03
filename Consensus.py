import csv
import os.path

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

        sorted(percent_list, key=lambda percent_list: percent_list[0], reverse=True)
        print(percent_list)

    def __get_large_overhead(self, normal_path_list, attack_path_list):
        normal_cycles_list = []
        attack_cycles_list = []
        normal_instructions_list = []
        attack_instructions_list = []
        normal_branch_list = []
        attack_branch_list = []

        for path in normal_path_list:  # normal
            if path.find('_record_') > 0:
                if path.find(Constant.LIST_SEQUENCE[0]) > 0:  # cycles
                    normal_cycles_list = self.__read_csv(path)
                elif path.find(Constant.LIST_SEQUENCE[1]) > 0:  # instructions
                    normal_instructions_list = self.__read_csv(path)
                elif path.find(Constant.LIST_SEQUENCE[2]) > 0:  # branch
                    normal_branch_list = self.__read_csv(path)

        for path in attack_path_list:  # attack
            if path.find('_record_') > 0:
                if path.find(Constant.LIST_SEQUENCE[0]) > 0:  # cycles
                    attack_cycles_list = self.__read_csv(path)
                elif path.find(Constant.LIST_SEQUENCE[1]) > 0:  # instructions
                    attack_instructions_list = self.__read_csv(path)
                elif path.find(Constant.LIST_SEQUENCE[2]) > 0:  # branch
                    attack_branch_list = self.__read_csv(path)

        self.__get_overhead_diff(normal_cycles_list, attack_cycles_list)

    def load_dataset(self, root_path):
        normal_path = root_path + '/' + Constant.NORMAL
        attack_path = root_path + '/' + Constant.ATTACK
        normal_path_list = self.__get_file_list_in_dir(normal_path)
        attack_path_list = self.__get_file_list_in_dir(attack_path)

        self.__get_large_overhead(normal_path_list, attack_path_list)
