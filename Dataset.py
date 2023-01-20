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
        temp_line_list = []
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

        self.__parsing_data(normal_gs_record_list, normal_gs_stat_list, normal_gs_top_list, normal_cs_record_list,
                            normal_cs_stat_list, normal_cs_top_list)

    def __parsing_data(self, normal_gs_record_list, normal_gs_stat_list, normal_gs_top_list, normal_cs_record_list,
                       normal_cs_stat_list, normal_cs_top_list):
        for normal_gs_record_sub_list in normal_gs_record_list:
            for normal_gs_record in normal_gs_record_sub_list:
                if len(normal_gs_record) > 0:
                    if normal_gs_record[0] != '#':
                        if normal_gs_record.find('[k]') > -1:
                            print(normal_gs_record)
