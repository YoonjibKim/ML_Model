import csv
import json
import os
import Constant


class Authentication_Time_Feature_Engineering:
    def __init__(self, scenario, random_cs_mode, gaussian_mode):
        root_path = Constant.BEFORE_PARSING_DATASET_PATH
        default_path = root_path + '/' + scenario + '/' + random_cs_mode + '/' + gaussian_mode
        attack_path = default_path + '/' + Constant.ATTACK
        normal_path = default_path + '/' + Constant.NORMAL

        attack_cs_id_pid_list_path = attack_path + '/' + Constant.CS_ID_PID_FILE_NANE
        normal_cs_id_pid_list_path = normal_path + '/' + Constant.CS_ID_PID_FILE_NANE

        attack_cs_id_pid_list = self.__read_csv(attack_cs_id_pid_list_path)
        normal_cs_id_pid_list = self.__read_csv(normal_cs_id_pid_list_path)
        attack_cs_id_pid_list = attack_cs_id_pid_list[1:]
        normal_cs_id_pid_list = normal_cs_id_pid_list[1:]

        attack_cs_record_path = attack_path + '/' + Constant.CS_RECORD
        normal_cs_record_path = normal_path + '/' + Constant.CS_RECORD

        attack_file_list = self.__get_file_names(attack_cs_record_path)
        normal_file_list = self.__get_file_names(normal_cs_record_path)

        attack_cs_pid_list = self.__get_cs_id_from_file_name_list(attack_file_list)
        normal_cs_pid_list = self.__get_cs_id_from_file_name_list(normal_file_list)
        chosen_attack_cs_id_list = self.__match_cs_pid_to_id(attack_cs_pid_list, attack_cs_id_pid_list)
        chosen_normal_cs_id_list = self.__match_cs_pid_to_id(normal_cs_pid_list, normal_cs_id_pid_list)
        intersection_cs_id_list = set(chosen_normal_cs_id_list) & set(chosen_attack_cs_id_list)

        attack_time_diff_path = attack_path + '/' + Constant.ATTACK_TIME_DIFF_FILE_NANE
        normal_time_diff_path = normal_path + '/' + Constant.NORMAL_TIME_DIFF_FILE_NAME
        attack_time_diff_dict = self.__read_json_txt_data(attack_time_diff_path)
        normal_time_diff_dict = self.__read_json_txt_data(normal_time_diff_path)

        print('test')

    def __get_target_cs_authentication_time_diff(self, attack_time_diff_dict, normal_time_diff_dict,
                                                 intersection_cs_id_list):
        attack_temp_list = []
        normal_temp_list = []

        for cs_id in intersection_cs_id_list:
            attack_temp_list.extend(attack_time_diff_dict[cs_id])



    @classmethod
    def __read_json_txt_data(cls, path):
        with open(path, 'r') as file:
            data = json.load(file)

        return data

    def __match_cs_pid_to_id(self, cs_pid_list, cs_id_pid_list):
        chosen_cs_id_list = []
        for cs_pid in cs_pid_list:
            for cs_id_pid in cs_id_pid_list:
                _cs_pid = cs_id_pid[1]
                if cs_pid == _cs_pid:
                    chosen_cs_id_list.append(cs_id_pid[0])
                    break

        return chosen_cs_id_list

    def __get_cs_id_from_file_name_list(self, file_name_list):
        cs_id_list = []
        for file_name in file_name_list:
            temp_list = file_name.split('_')
            temp = temp_list[len(temp_list) - 1]
            temp_list = temp.split('.')
            cs_id = temp_list[0]
            cs_id_list.append(cs_id)

        return cs_id_list

    @classmethod
    def __get_file_names(cls, path):
        file_name_list = []
        _file_name_list = os.listdir(path)
        for file_name in _file_name_list:
            if file_name.find('.txt') > 0:
                file_name_list.append(file_name)

        return file_name_list

    @classmethod
    def __read_csv(cls, path):
        read_line = []
        with open(path, 'r') as f:
            rdr = csv.reader(f)
            for line in rdr:
                read_line.append(line)

        return read_line
