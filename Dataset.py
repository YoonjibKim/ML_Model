import copy
import json
import os
import Constant


class Dataset:
    __normal_time_diff = {}
    __normal_gs_record_list = []
    __normal_gs_stat_list = []
    __normal_gs_top_list = []
    __normal_cs_record_list = []
    __normal_cs_stat_list = []
    __normal_cs_top_list = []

    __attack_time_diff = {}
    __attack_gs_record_list = []
    __attack_gs_stat_list = []
    __attack_gs_top_list = []
    __attack_cs_record_list = []
    __attack_cs_stat_list = []
    __attack_cs_top_list = []

    @classmethod
    def __get_file_names(cls, path):
        file_name_list = []
        _file_name_list = os.listdir(path)
        for file_name in _file_name_list:
            if file_name.find('.txt') > 0:
                file_name_list.append(file_name)

        return file_name_list

    @classmethod
    def __read_txt_data(cls, path):
        with open(path, 'r') as fd:
            temp_line_list = fd.readlines()

        line_list = []
        for line in temp_line_list:
            line = line.strip()
            line_list.append(line)
        return line_list

    @classmethod
    def __read_json_data(cls, path):
        with open(path, 'r') as fd:
            data_dict = json.load(fd)

        return data_dict

    @classmethod
    def __generate_full_path(cls, _dir, file_name_list):
        full_path_list = []
        for file_name in file_name_list:
            temp = _dir + '/' + file_name
            full_path_list.append(temp)

        return full_path_list

    @classmethod
    def __read_files(cls, time_diff_path, gs_record_path_list, gs_stat_path_list, gs_top_path_list,
                     cs_record_path_list, cs_stat_path_list, cs_top_path_list):
        time_diff_file = cls.__read_json_data(time_diff_path)
        gs_record_file_list = []
        for gs_record_path in gs_record_path_list:
            gs_record_file = cls.__read_txt_data(gs_record_path)
            gs_record_file_list.append(gs_record_file)

        gs_stat_file_list = []
        for gs_stat_path in gs_stat_path_list:
            gs_stat_file = cls.__read_txt_data(gs_stat_path)
            gs_stat_file_list.append(gs_stat_file)

        gs_top_file_list = []
        for gs_top_path in gs_top_path_list:
            gs_top_file = cls.__read_txt_data(gs_top_path)
            gs_top_file_list.append(gs_top_file)

        cs_record_file_list = []
        for cs_record_path in cs_record_path_list:
            cs_record_file = cls.__read_txt_data(cs_record_path)
            cs_record_file_list.append(cs_record_file)

        cs_stat_file_list = []
        for cs_stat_path in cs_stat_path_list:
            cs_stat_file = cls.__read_txt_data(cs_stat_path)
            cs_stat_file_list.append(cs_stat_file)

        cs_top_file_list = []
        for cs_top_path in cs_top_path_list:
            cs_top_file = cls.__read_txt_data(cs_top_path)
            cs_top_file_list.append(cs_top_file)

        return time_diff_file, gs_record_file_list, gs_stat_file_list, gs_top_file_list, cs_record_file_list, \
            cs_stat_file_list, cs_top_file_list

    def access_dataset(self, attack_scenario, random_cs, gaussian):
        base_path = Constant.ROOT_INPUT_DIR_NAME + '/' + attack_scenario + '/' + random_cs + '/' + gaussian

        attack_path = base_path + '/' + Constant.ATTACK
        attack_gs_record_path = attack_path + '/' + Constant.GS_RECORD
        attack_gs_stat_path = attack_path + '/' + Constant.GS_STAT
        attack_gs_top_path = attack_path + '/' + Constant.GS_TOP
        attack_cs_record_path = attack_path + '/' + Constant.CS_RECORD
        attack_cs_stat_path = attack_path + '/' + Constant.CS_STAT
        attack_cs_top_path = attack_path + '/' + Constant.CS_TOP

        attack_gs_record_file_name_list = self.__get_file_names(attack_gs_record_path)
        attack_gs_stat_file_name_list = self.__get_file_names(attack_gs_stat_path)
        attack_gs_top_file_name_list = self.__get_file_names(attack_gs_top_path)
        attack_cs_record_file_name_list = self.__get_file_names(attack_cs_record_path)
        attack_cs_stat_file_name_list = self.__get_file_names(attack_cs_stat_path)
        attack_cs_top_file_name_list = self.__get_file_names(attack_cs_top_path)

        attack_time_diff_path = attack_path + '/' + 'attack_time_diff.txt'
        attack_gs_record_path_list = self.__generate_full_path(attack_gs_record_path, attack_gs_record_file_name_list)
        attack_gs_stat_path_list = self.__generate_full_path(attack_gs_stat_path, attack_gs_stat_file_name_list)
        attack_gs_top_path_list = self.__generate_full_path(attack_gs_top_path, attack_gs_top_file_name_list)
        attack_cs_record_path_list = self.__generate_full_path(attack_cs_record_path, attack_cs_record_file_name_list)
        attack_cs_stat_path_list = self.__generate_full_path(attack_cs_stat_path, attack_cs_stat_file_name_list)
        attack_cs_top_path_list = self.__generate_full_path(attack_cs_top_path, attack_cs_top_file_name_list)

        normal_path = base_path + '/' + Constant.NORMAL
        normal_gs_record_path = normal_path + '/' + Constant.GS_RECORD
        normal_gs_stat_path = normal_path + '/' + Constant.GS_STAT
        normal_gs_top_path = normal_path + '/' + Constant.GS_TOP
        normal_cs_record_path = normal_path + '/' + Constant.CS_RECORD
        normal_cs_stat_path = normal_path + '/' + Constant.CS_STAT
        normal_cs_top_path = normal_path + '/' + Constant.CS_TOP

        normal_gs_record_file_name_list = self.__get_file_names(normal_gs_record_path)
        normal_gs_stat_file_name_list = self.__get_file_names(normal_gs_stat_path)
        normal_gs_top_file_name_list = self.__get_file_names(normal_gs_top_path)
        normal_cs_record_file_name_list = self.__get_file_names(normal_cs_record_path)
        normal_cs_stat_file_name_list = self.__get_file_names(normal_cs_stat_path)
        normal_cs_top_file_name_list = self.__get_file_names(normal_cs_top_path)

        normal_time_diff_path = normal_path + '/' + 'normal_time_diff.txt'
        normal_gs_record_path_list = self.__generate_full_path(normal_gs_record_path, normal_gs_record_file_name_list)
        normal_gs_stat_path_list = self.__generate_full_path(normal_gs_stat_path, normal_gs_stat_file_name_list)
        normal_gs_top_path_list = self.__generate_full_path(normal_gs_top_path, normal_gs_top_file_name_list)
        normal_cs_record_path_list = self.__generate_full_path(normal_cs_record_path, normal_cs_record_file_name_list)
        normal_cs_stat_path_list = self.__generate_full_path(normal_cs_stat_path, normal_cs_stat_file_name_list)
        normal_cs_top_path_list = self.__generate_full_path(normal_cs_top_path, normal_cs_top_file_name_list)

        normal_time_diff, normal_gs_record_list, normal_gs_stat_list, normal_gs_top_list, normal_cs_record_list, \
            normal_cs_stat_list, normal_cs_top_list = \
            self.__read_files(normal_time_diff_path, normal_gs_record_path_list, normal_gs_stat_path_list,
                              normal_gs_top_path_list, normal_cs_record_path_list, normal_cs_stat_path_list,
                              normal_cs_top_path_list)

        attack_time_diff, attack_gs_record_list, attack_gs_stat_list, attack_gs_top_list, attack_cs_record_list, \
            attack_cs_stat_list, attack_cs_top_list = \
            self.__read_files(attack_time_diff_path, attack_gs_record_path_list, attack_gs_stat_path_list,
                              attack_gs_top_path_list, attack_cs_record_path_list, attack_cs_stat_path_list,
                              attack_cs_top_path_list)

        normal_gs_top_file_and_path_list = [normal_gs_top_path_list, normal_gs_top_list]
        normal_cs_top_file_and_path_list = [normal_cs_top_path_list, normal_cs_top_list]

        gs_normal_record_list, gs_normal_stat_list, gs_normal_total_top_sec_measure_list, cs_normal_record_list, \
            cs_normal_stat_list, cs_normal_total_top_sec_measure_list = \
            self.__loading_data(normal_gs_record_list, normal_gs_stat_list, normal_gs_top_file_and_path_list,
                                normal_cs_record_list, normal_cs_stat_list, normal_cs_top_file_and_path_list)

        attack_gs_top_file_and_path_list = [attack_gs_top_path_list, attack_gs_top_list]
        attack_cs_top_file_and_path_list = [attack_cs_top_path_list, attack_cs_top_list]

        gs_attack_record_list, gs_attack_stat_list, gs_attack_total_top_sec_measure_list, cs_attack_record_list, \
            cs_attack_stat_list, cs_attack_total_top_sec_measure_list = \
            self.__loading_data(attack_gs_record_list, attack_gs_stat_list, attack_gs_top_file_and_path_list,
                                attack_cs_record_list, attack_cs_stat_list, attack_cs_top_file_and_path_list)

        self.__normal_time_diff = copy.deepcopy(normal_time_diff)
        self.__normal_gs_record_list = copy.deepcopy(gs_normal_record_list)
        self.__normal_gs_top_list = copy.deepcopy(gs_normal_total_top_sec_measure_list)
        self.__normal_gs_stat_list = copy.deepcopy(gs_normal_stat_list)
        self.__normal_cs_record_list = copy.deepcopy(cs_normal_record_list)
        self.__normal_cs_top_list = copy.deepcopy(cs_normal_total_top_sec_measure_list)
        self.__normal_cs_stat_list = copy.deepcopy(cs_normal_stat_list)

        self.__attack_time_diff = copy.deepcopy(attack_time_diff)
        self.__attack_gs_record_list = copy.deepcopy(gs_attack_record_list)
        self.__attack_gs_top_list = copy.deepcopy(gs_attack_total_top_sec_measure_list)
        self.__attack_gs_stat_list = copy.deepcopy(gs_attack_stat_list)
        self.__attack_cs_record_list = copy.deepcopy(cs_attack_record_list)
        self.__attack_cs_top_list = copy.deepcopy(cs_attack_total_top_sec_measure_list)
        self.__attack_cs_stat_list = copy.deepcopy(cs_attack_stat_list)

        print('test')

    @classmethod
    def __get_gs_top_file(cls, top_file_and_path_list):
        path_list = top_file_and_path_list[0]
        file_list = top_file_and_path_list[1]
        top_cycle_sec_measure_list = []
        top_branch_sec_measure_list = []
        top_inst_sec_measure_list = []
        for index, path in enumerate(path_list):
            file = file_list[index]
            if path.find('perf_top_cycles') > -1:
                shallow_list = []
                for sub_file in file:
                    if len(sub_file) > 0:
                        if sub_file.find('[k]') > -1:
                            sub_file = sub_file.split()
                            sub_file = sub_file[0] + ' ' + sub_file[4]
                            shallow_list.append(sub_file)
                        elif sub_file.find('-------') > -1:
                            if len(shallow_list) > 0:
                                top_cycle_sec_measure_list.append(copy.deepcopy(shallow_list))
                                shallow_list.clear()
            elif path.find('perf_top_instructions') > -1:
                shallow_list = []
                for sub_file in file:
                    if len(sub_file) > 0:
                        if sub_file.find('[k]') > -1:
                            sub_file = sub_file.split()
                            sub_file = sub_file[0] + ' ' + sub_file[4]
                            shallow_list.append(sub_file)
                        elif sub_file.find('-------') > -1:
                            if len(shallow_list) > 0:
                                top_inst_sec_measure_list.append(copy.deepcopy(shallow_list))
                                shallow_list.clear()
            elif path.find('perf_top_branch') > -1:
                shallow_list = []
                for sub_file in file:
                    if len(sub_file) > 0:
                        if sub_file.find('[k]') > -1:
                            sub_file = sub_file.split()
                            sub_file = sub_file[0] + ' ' + sub_file[4]
                            shallow_list.append(sub_file)
                        elif sub_file.find('-------') > -1:
                            if len(shallow_list) > 0:
                                top_branch_sec_measure_list.append(copy.deepcopy(shallow_list))
                                shallow_list.clear()
        total_top_sec_measure_list = [top_cycle_sec_measure_list, top_inst_sec_measure_list,
                                      top_branch_sec_measure_list]
        return copy.deepcopy(total_top_sec_measure_list)

    def __loading_data(self, gs_record_list, gs_stat_list, gs_top_file_and_path_list, cs_record_list,
                       cs_stat_list, cs_top_file_and_path_list):
        gs_record_cycle_list = []
        gs_record_branch_list = []
        gs_record_instructions_list = []
        gs_cycle_flag = False
        gs_branch_flag = False
        gs_instruction_flag = False
        for gs_record_sub_list in gs_record_list:
            for gs_record in gs_record_sub_list:
                if len(gs_record) > 0:
                    if gs_record.find("'cycles'") > -1:
                        gs_cycle_flag = True
                        gs_branch_flag = False
                        gs_instruction_flag = False
                    elif gs_record.find("'branch'") > -1:
                        gs_cycle_flag = False
                        gs_branch_flag = True
                        gs_instruction_flag = False
                    elif gs_record.find("'instructions'") > -1:
                        gs_cycle_flag = False
                        gs_branch_flag = False
                        gs_instruction_flag = True

                    if gs_cycle_flag:
                        if gs_record.find('[k]') > -1:
                            temp_list = gs_record.split()
                            elements = [temp_list[0], temp_list[4]]
                            gs_record_cycle_list.append(elements)
                    elif gs_branch_flag:
                        if gs_record.find('[k]') > -1:
                            temp_list = gs_record.split()
                            elements = [temp_list[0], temp_list[4]]
                            gs_record_branch_list.append(elements)
                    elif gs_instruction_flag:
                        if gs_record.find('[k]') > -1:
                            temp_list = gs_record.split()
                            elements = [temp_list[0], temp_list[4]]
                            gs_record_instructions_list.append(elements)

        gs_stat_cycle_list = []
        gs_stat_instruction_list = []
        gs_stat_branch_list = []
        for gs_stat_sub_list in gs_stat_list:
            for gs_stat in gs_stat_sub_list:
                if len(gs_stat) > 0:
                    if gs_stat[0] != '#':
                        temp_gs_stat_list = gs_stat.split()
                        if '<not' not in temp_gs_stat_list:
                            if 'cycles' in temp_gs_stat_list:
                                counts = temp_gs_stat_list[1]
                                gs_stat_cycle_list.append(counts)
                            elif 'instructions' in temp_gs_stat_list:
                                counts = temp_gs_stat_list[1]
                                gs_stat_instruction_list.append(counts)
                            elif 'branches' in temp_gs_stat_list:
                                counts = temp_gs_stat_list[1]
                                gs_stat_branch_list.append(counts)

        gs_total_top_sec_measure_list = self.__get_gs_top_file(gs_top_file_and_path_list)

        _cs_record_list = []
        for cs_record_sub_list in cs_record_list:
            cs_record_cycle_list = []
            cs_record_instruction_list = []
            cs_record_branch_list = []
            for cs_record in cs_record_sub_list:
                if len(cs_record) > 0:
                    if cs_record.find("'cycles'") > -1:
                        gs_cycle_flag = True
                        gs_branch_flag = False
                        gs_instruction_flag = False
                    elif cs_record.find("'branch'") > -1:
                        gs_cycle_flag = False
                        gs_branch_flag = True
                        gs_instruction_flag = False
                    elif cs_record.find("'instructions'") > -1:
                        gs_cycle_flag = False
                        gs_branch_flag = False
                        gs_instruction_flag = True

                    if gs_cycle_flag:
                        if cs_record.find('[k]') > -1:
                            temp_list = cs_record.split()
                            elements = [temp_list[0], temp_list[4]]
                            cs_record_cycle_list.append(elements)
                    elif gs_branch_flag:
                        if cs_record.find('[k]') > -1:
                            temp_list = cs_record.split()
                            elements = [temp_list[0], temp_list[4]]
                            cs_record_branch_list.append(elements)
                    elif gs_instruction_flag:
                        if cs_record.find('[k]') > -1:
                            temp_list = cs_record.split()
                            elements = [temp_list[0], temp_list[4]]
                            cs_record_instruction_list.append(elements)

            _cs_record_list.append([cs_record_cycle_list, cs_record_instruction_list, cs_record_branch_list])

        _cs_stat_list = []
        for cs_stat_sub_list in cs_stat_list:
            cs_stat_cycle_list = []
            cs_stat_instruction_list = []
            cs_stat_branch_list = []
            for cs_stat in cs_stat_sub_list:
                if len(cs_stat) > 0:
                    if cs_stat[0] != '#':
                        temp_cs_stat_list = cs_stat.split()
                        if '<not' not in temp_cs_stat_list:
                            if 'cycles' in temp_cs_stat_list:
                                counts = temp_cs_stat_list[1]
                                cs_stat_cycle_list.append(counts)
                            elif 'instructions' in temp_cs_stat_list:
                                counts = temp_cs_stat_list[1]
                                cs_stat_instruction_list.append(counts)
                            elif 'branches' in temp_cs_stat_list:
                                counts = temp_cs_stat_list[1]
                                cs_stat_branch_list.append(counts)

            _cs_stat_list.append([cs_stat_cycle_list, cs_stat_instruction_list, cs_stat_branch_list])

        path_list = cs_top_file_and_path_list[0]
        file_list = cs_top_file_and_path_list[1]

        cycle_path_file_list = []
        branch_path_file_list = []
        instruction_path_file_list = []
        cycle_name = 'perf_top_cycles'
        instruction_name = 'perf_top_instructions'
        branch_name = 'perf_top_branch'
        for index, path in enumerate(path_list):
            if path.find(cycle_name) > -1:
                cycle_path_file_list.append([path, file_list[index]])
            elif path.find(instruction_name) > -1:
                instruction_path_file_list.append([path, file_list[index]])
            elif path.find(branch_name) > -1:
                branch_path_file_list.append([path, file_list[index]])

        cycle_path_file_list.sort()
        branch_path_file_list.sort()
        instruction_path_file_list.sort()

        cs_each_top_list = []
        for index in range(0, len(cycle_path_file_list)):
            temp_cycle_path = cycle_path_file_list[index][0]
            temp_instruction_path = instruction_path_file_list[index][0]
            temp_branch_path = branch_path_file_list[index][0]

            temp_cycle_file = cycle_path_file_list[index][1]
            temp_instruction_file = instruction_path_file_list[index][1]
            temp_branch_file = branch_path_file_list[index][1]

            temp_path_list = [temp_cycle_path, temp_instruction_path, temp_branch_path]
            temp_file_list = [temp_cycle_file, temp_instruction_file, temp_branch_file]

            cs_total_top_list = [temp_path_list, temp_file_list]
            cs_each_top_list.append(cs_total_top_list)

        cs_total_top_sec_measure_list = []
        for cs_top_list in cs_each_top_list:
            cs_top_sec_measure_list = self.__get_gs_top_file(cs_top_list)
            cs_total_top_sec_measure_list.append(cs_top_sec_measure_list)

        ret_gs_record_list = [gs_record_cycle_list, gs_record_instructions_list, gs_record_branch_list]
        ret_gs_stat_list = [gs_stat_cycle_list, gs_stat_instruction_list, gs_stat_branch_list]

        return copy.deepcopy(ret_gs_record_list), copy.deepcopy(ret_gs_stat_list), \
            copy.deepcopy(gs_total_top_sec_measure_list), copy.deepcopy(_cs_record_list), \
            copy.deepcopy(_cs_stat_list), copy.deepcopy(cs_total_top_sec_measure_list)
