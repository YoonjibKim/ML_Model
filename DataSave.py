import csv
import json
import os
import Constant


class DataSave:
    def __init__(self):
        print('data save')

    @classmethod
    def __create_directory(cls, dir_path):
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except OSError:
            print('Error: Creating directory. ' + dir_path)

    @classmethod
    def __get_file_name(cls, file_path):
        temp_path_list = file_path.split('/')
        file_name = temp_path_list[len(temp_path_list) - 1]
        return file_name

    @classmethod
    def save_gs_features_to_storage(cls, dataset):
        save_dir_path = dataset.get_base_dir_path()
        save_normal_path = save_dir_path + '/' + Constant.NORMAL
        save_attack_path = save_dir_path + '/' + Constant.ATTACK
        cls.__create_directory(save_normal_path)
        cls.__create_directory(save_attack_path)

        file_name = cls.__get_file_name(dataset.get_normal_time_diff_path())
        file_path = save_normal_path + '/' + file_name
        with open(file_path, 'w') as fd:
            json.dump(dataset.get_normal_time_diff_file(), fd)

        file_name = cls.__get_file_name(dataset.get_attack_time_diff_path())
        file_path = save_attack_path + '/' + file_name
        with open(file_path, 'w') as fd:
            json.dump(dataset.get_attack_time_diff_file(), fd)

        list_sequence = ['cycles', 'instructions', 'branch']
        for index, file_list in enumerate(dataset.get_normal_gs_record_file_list()):
            save_path = save_normal_path + '/gs_record_' + list_sequence[index] + '.csv'
            with open(save_path, 'w', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerows(file_list)

        for index, file_list in enumerate(dataset.get_attack_gs_record_file_list()):
            save_path = save_attack_path + '/gs_record_' + list_sequence[index] + '.csv'
            with open(save_path, 'w', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerows(file_list)

        for index, file_list in enumerate(dataset.get_normal_gs_stat_file_list()):
            save_path = save_normal_path + '/gs_stat_' + list_sequence[index] + '.txt'
            with open(save_path, 'w', newline='') as fd:
                for record in file_list:
                    fd.write(record + '\n')

        for index, file_list in enumerate(dataset.get_attack_gs_stat_file_list()):
            save_path = save_attack_path + '/gs_stat_' + list_sequence[index] + '.txt'
            with open(save_path, 'w', newline='') as fd:
                for record in file_list:
                    fd.write(record + '\n')

        file_list = dataset.get_normal_gs_top_file_list()
        normal_gs_top_path_list = dataset.get_normal_gs_top_path_list()
        for index, path in enumerate(normal_gs_top_path_list):
            save_path = save_normal_path + '/gs_top_' + list_sequence[index] + '.csv'
            with open(save_path, 'w', newline='') as fd:
                writer = csv.writer(fd)
                for record in file_list[index]:
                    writer.writerow(record)

        file_list = dataset.get_attack_gs_top_file_list()
        attack_gs_top_path_list = dataset.get_attack_gs_top_path_list()
        for index, path in enumerate(attack_gs_top_path_list):
            save_path = save_attack_path + '/gs_top_' + list_sequence[index] + '.csv'
            with open(save_path, 'w', newline='') as fd:
                writer = csv.writer(fd)
                for record in file_list[index]:
                    writer.writerow(record)

    @classmethod
    def save_cs_features_to_storage(cls, dataset):
        save_dir_path = dataset.get_base_dir_path()
        save_normal_path = save_dir_path + '/' + Constant.NORMAL
        save_attack_path = save_dir_path + '/' + Constant.ATTACK
        cls.__create_directory(save_normal_path)
        cls.__create_directory(save_attack_path)
        list_sequence = ['cycles', 'instructions', 'branch']

        normal_cs_record_path_list = dataset.get_normal_cs_record_path_list()
        normal_cs_file_name_list = []
        for normal_cs_record_path in normal_cs_record_path_list:
            temp_list = normal_cs_record_path.split('/')
            file_name = temp_list[len(temp_list) - 1]
            normal_cs_file_name_list.append(file_name)

        normal_cs_record_file_list = dataset.get_normal_cs_record_file_list()
        for index_i, file_name in enumerate(normal_cs_file_name_list):
            for index_j, normal_cs_record_file in enumerate(normal_cs_record_file_list[index_i]):
                _file_name = file_name.replace('perf_record', 'cs_record_' + list_sequence[index_j])
                _file_name = _file_name.replace('.txt', '.csv')
                save_path = save_normal_path + '/' + _file_name
                with open(save_path, 'w', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerows(normal_cs_record_file)

        attack_cs_record_path_list = dataset.get_attack_cs_record_path_list()
        attack_cs_file_name_list = []
        for attack_cs_record_path in attack_cs_record_path_list:
            temp_list = attack_cs_record_path.split('/')
            file_name = temp_list[len(temp_list) - 1]
            attack_cs_file_name_list.append(file_name)

        attack_cs_record_file_list = dataset.get_attack_cs_record_file_list()
        for index_i, file_name in enumerate(attack_cs_file_name_list):
            for index_j, attack_cs_record_file in enumerate(attack_cs_record_file_list[index_i]):
                _file_name = file_name.replace('perf_record', 'cs_record_' + list_sequence[index_j])
                _file_name = _file_name.replace('.txt', '.csv')
                save_path = save_attack_path + '/' + _file_name
                with open(save_path, 'w', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerows(attack_cs_record_file)

        normal_cs_stat_file_list = dataset.get_normal_cs_stat_file_list()
        normal_cs_stat_path = dataset.get_normal_cs_stat_path_list()
        for i in normal_cs_stat_file_list:
            print(i)

