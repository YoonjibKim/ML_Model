import csv
import json
import os

import numpy as np
import pandas as pd

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
    def __get_cs_id_from_pid(cls, file_name, cs_pid_id_list, extension):
        temp_list = file_name.split('_')
        cs_pid_from_file_name = temp_list[len(temp_list) - 1]
        cs_pid_from_file_name = cs_pid_from_file_name.split('.')
        cs_pid_from_file_name = cs_pid_from_file_name[0]

        cs_id = None

        for record in cs_pid_id_list:
            cs_pid = record[1]
            if cs_pid_from_file_name == cs_pid:
                cs_id = record[0]
                break

        new_file_name = ''
        for temp in temp_list[:-1]:
            new_file_name += temp
            new_file_name += '_'

        return new_file_name + cs_id + '.' + extension

    @classmethod
    def save_gs_top_features_to_storage(cls, dataset):
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

        for index, file_list in enumerate(dataset.get_normal_gs_record_file_list()):
            save_path = save_normal_path + '/gs_record_' + Constant.LIST_SEQUENCE[index] + '.csv'
            with open(save_path, 'w', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerows(file_list)

        for index, file_list in enumerate(dataset.get_attack_gs_record_file_list()):
            save_path = save_attack_path + '/gs_record_' + Constant.LIST_SEQUENCE[index] + '.csv'
            with open(save_path, 'w', newline='') as fd:
                writer = csv.writer(fd)
                writer.writerows(file_list)

        for index, file_list in enumerate(dataset.get_normal_gs_stat_file_list()):
            save_path = save_normal_path + '/gs_stat_' + Constant.LIST_SEQUENCE[index] + '.txt'
            with open(save_path, 'w', newline='') as fd:
                for record in file_list:
                    fd.write(record + '\n')

        for index, file_list in enumerate(dataset.get_attack_gs_stat_file_list()):
            save_path = save_attack_path + '/gs_stat_' + Constant.LIST_SEQUENCE[index] + '.txt'
            with open(save_path, 'w', newline='') as fd:
                for record in file_list:
                    fd.write(record + '\n')

        file_list = dataset.get_normal_gs_top_file_list()
        normal_gs_top_path_list = dataset.get_normal_gs_top_path_list()
        for index, path in enumerate(normal_gs_top_path_list):
            save_path = save_normal_path + '/gs_top_' + Constant.LIST_SEQUENCE[index] + '.csv'
            with open(save_path, 'w', newline='') as fd:
                writer = csv.writer(fd)
                for record in file_list[index]:
                    writer.writerow(record)

        file_list = dataset.get_attack_gs_top_file_list()
        attack_gs_top_path_list = dataset.get_attack_gs_top_path_list()
        for index, path in enumerate(attack_gs_top_path_list):
            save_path = save_attack_path + '/gs_top_' + Constant.LIST_SEQUENCE[index] + '.csv'
            with open(save_path, 'w', newline='') as fd:
                writer = csv.writer(fd)
                for record in file_list[index]:
                    writer.writerow(record)

    @classmethod
    def save_cs_top_features_to_storage(cls, dataset):
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
            file_path = temp_list[len(temp_list) - 1]
            normal_cs_file_name_list.append(file_path)

        normal_cs_record_file_list = dataset.get_normal_cs_record_file_list()
        for index_i, file_path in enumerate(normal_cs_file_name_list):
            for index_j, normal_cs_record_file in enumerate(normal_cs_record_file_list[index_i]):
                _file_name = file_path.replace('perf_record', 'cs_record_' + list_sequence[index_j])
                _file_name = cls.__get_cs_id_from_pid(_file_name, dataset.get_cs_id_pid_list(), 'csv')

                save_path = save_normal_path + '/' + _file_name
                with open(save_path, 'w', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerows(normal_cs_record_file)

        attack_cs_record_path_list = dataset.get_attack_cs_record_path_list()
        attack_cs_file_name_list = []
        for attack_cs_record_path in attack_cs_record_path_list:
            temp_list = attack_cs_record_path.split('/')
            file_path = temp_list[len(temp_list) - 1]
            attack_cs_file_name_list.append(file_path)

        attack_cs_record_file_list = dataset.get_attack_cs_record_file_list()
        for index_i, file_path in enumerate(attack_cs_file_name_list):
            for index_j, attack_cs_record_file in enumerate(attack_cs_record_file_list[index_i]):
                _file_name = file_path.replace('perf_record', 'cs_record_' + list_sequence[index_j])
                _file_name = cls.__get_cs_id_from_pid(_file_name, dataset.get_cs_id_pid_list(), 'csv')
                save_path = save_attack_path + '/' + _file_name
                with open(save_path, 'w', newline='') as fd:
                    writer = csv.writer(fd)
                    writer.writerows(attack_cs_record_file)

        normal_cs_stat_file_list = dataset.get_normal_cs_stat_file_list()
        normal_cs_stat_path_list = dataset.get_normal_cs_stat_path_list()
        for index_i, file_path in enumerate(normal_cs_stat_path_list):
            temp_file_name_list = file_path.split('/')
            _file_name = temp_file_name_list[len(temp_file_name_list) - 1]
            for index_j, file in enumerate(normal_cs_stat_file_list[index_i]):
                sequence = list_sequence[index_j]
                __file_name = _file_name.replace('perf_stat_', 'cs_stat_' + sequence + '_')
                __file_name = cls.__get_cs_id_from_pid(__file_name, dataset.get_cs_id_pid_list(), 'txt')
                save_path = save_normal_path + '/' + __file_name
                with open(save_path, 'w', newline='') as fd:
                    for record in file:
                        fd.write(record + '\n')

        attack_cs_stat_file_list = dataset.get_attack_cs_stat_file_list()
        attack_cs_stat_path_list = dataset.get_attack_cs_stat_path_list()
        for index_i, file_path in enumerate(attack_cs_stat_path_list):
            temp_file_name_list = file_path.split('/')
            _file_name = temp_file_name_list[len(temp_file_name_list) - 1]
            for index_j, file in enumerate(attack_cs_stat_file_list[index_i]):
                sequence = list_sequence[index_j]
                __file_name = _file_name.replace('perf_stat_', 'cs_stat_' + sequence + '_')
                __file_name = cls.__get_cs_id_from_pid(__file_name, dataset.get_cs_id_pid_list(), 'txt')
                save_path = save_attack_path + '/' + __file_name
                with open(save_path, 'w', newline='') as fd:
                    for record in file:
                        fd.write(record + '\n')

        normal_cs_top_file_list = dataset.get_normal_cs_top_file_list()
        normal_cs_top_path_list = dataset.get_normal_cs_top_path_list()
        for index_i, file_path_list in enumerate(normal_cs_top_path_list):
            for index_j, file_path in enumerate(file_path_list):
                temp_file_name_list = file_path.split('/')
                _file_name = temp_file_name_list[len(temp_file_name_list) - 1]
                __file_name = _file_name.replace('perf_top', 'cs_top')
                __file_name = cls.__get_cs_id_from_pid(__file_name, dataset.get_cs_id_pid_list(), 'txt')
                save_path = save_normal_path + '/' + __file_name
                data_list = normal_cs_top_file_list[index_i][index_j]
                symbol_name_list = []
                organized_record_list = []
                for record_list in data_list:
                    for record in record_list:
                        symbol_name = record.split()[1]
                        symbol_name_list.append(symbol_name)
                        organized_record_list.append(record.split())

                unique_symbol_name_list = list(set(symbol_name_list))
                symbol_name_overhead_dict = {}
                for symbol_name in unique_symbol_name_list:
                    temp_list = []
                    for record in organized_record_list:
                        if symbol_name in record:
                            temp_list.append(record[0])
                    symbol_name_overhead_dict[symbol_name] = temp_list

                with open(save_path, 'w', newline='') as f:
                    for symbol_name, overhead_list in symbol_name_overhead_dict.items():
                        f.write(symbol_name + ' ')
                        for overhead in overhead_list:
                            overhead = overhead[:-1]
                            f.write(overhead + ' ')
                        f.write('\n')

        attack_cs_top_file_list = dataset.get_attack_cs_top_file_list()
        attack_cs_top_path_list = dataset.get_attack_cs_top_path_list()
        for index_i, file_path_list in enumerate(attack_cs_top_path_list):
            for index_j, file_path in enumerate(file_path_list):
                temp_file_name_list = file_path.split('/')
                _file_name = temp_file_name_list[len(temp_file_name_list) - 1]
                __file_name = _file_name.replace('perf_top', 'cs_top')
                __file_name = cls.__get_cs_id_from_pid(__file_name, dataset.get_cs_id_pid_list(), 'txt')
                save_path = save_attack_path + '/' + __file_name
                data_list = attack_cs_top_file_list[index_i][index_j]
                symbol_name_list = []
                organized_record_list = []
                for record_list in data_list:
                    for record in record_list:
                        symbol_name = record.split()[1]
                        symbol_name_list.append(symbol_name)
                        organized_record_list.append(record.split())

                unique_symbol_name_list = list(set(symbol_name_list))
                symbol_name_overhead_dict = {}
                for symbol_name in unique_symbol_name_list:
                    temp_list = []
                    for record in organized_record_list:
                        if symbol_name in record:
                            temp_list.append(record[0])
                    symbol_name_overhead_dict[symbol_name] = temp_list

                with open(save_path, 'w', newline='') as f:
                    for symbol_name, overhead_list in symbol_name_overhead_dict.items():
                        f.write(symbol_name + ' ')
                        for overhead in overhead_list:
                            overhead = overhead[:-1]
                            f.write(overhead + ' ')
                        f.write('\n')

    @classmethod
    def save_profiling_data(cls, gs_stat_diff_list, cs_stat_diff_list, gs_record_diff_list, cs_record_diff_list,
                            gs_record_diff_mean_std, cs_record_diff_mean_std):
        with open('./Output_results/mean_std_diff.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['gs arithmetic mean & gs std', 'type'])
            wr.writerows(gs_record_diff_mean_std)
            wr.writerows(['\n'])
            wr.writerow(['cs arithmetic mean & cs std', 'type'])
            wr.writerows(cs_record_diff_mean_std)

        with open('./Output_results/stat_diff.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['gs stat diff'])
            wr.writerows(gs_stat_diff_list)
            wr.writerow(['\n'])
            wr.writerow(['cs stat diff'])
            wr.writerows(cs_stat_diff_list)

        with open('./Output_results/gs_record_diff.csv', 'w') as f:
            wr = csv.writer(f)
            for sequence, gs_record_diff in zip(Constant.LIST_SEQUENCE, gs_record_diff_list):
                wr.writerow([sequence])
                wr.writerow(['overhead', 'symbol'])
                wr.writerows(gs_record_diff)
                wr.writerow('\n')

        with open('./Output_results/cs_record_diff.csv', 'w') as f:
            wr = csv.writer(f)
            for sequence, cs_record_diff in zip(Constant.LIST_SEQUENCE, cs_record_diff_list):
                wr.writerow([sequence])
                wr.writerow(['overhead', 'symbol'])
                wr.writerows(cs_record_diff)
                wr.writerow('\n')

    @classmethod
    def __remove_inner_list(cls, data_dict):
        converted_dict = {}
        size_list = []
        for symbol, data_array in data_dict.items():
            if len(np.shape(data_array)) > 1:
                temp_list = list(data[0] for data in data_array)
            else:
                temp_array = data_dict[symbol]
                temp_list = temp_array.tolist()

            converted_dict[symbol] = temp_list
            size_list.append(len(temp_list))

        size_list = sorted(size_list)
        min_size = size_list[0]

        ret_dict = {}
        for symbol, data_array in converted_dict.items():
            ret_dict[symbol] = data_array[:min_size]

        return ret_dict

    @classmethod
    def save_top_features(cls, training_normal_feature_dict, training_attack_feature_dict,
                          testing_normal_feature_dict, testing_attack_feature_dict, feature_type):
        if feature_type == Constant.EXTENDED_TOP_DATASET_PATH:
            temp_dict_1 = cls.__remove_inner_list(training_normal_feature_dict)
            training_normal_feature_df = pd.DataFrame(temp_dict_1)
            temp_dict_2 = cls.__remove_inner_list(training_attack_feature_dict)
            training_attack_feature_df = pd.DataFrame(temp_dict_2)
            temp_dict_3 = cls.__remove_inner_list(testing_normal_feature_dict)
            testing_normal_feature_df = pd.DataFrame(temp_dict_3)
            temp_dict_4 = cls.__remove_inner_list(testing_attack_feature_dict)
            testing_attack_feature_df = pd.DataFrame(temp_dict_4)

            training_normal_feature_df.to_csv(Constant.EXTENDED_TOP_DATASET_PATH + '/training_normal_feature.csv',
                                              index=False)
            training_attack_feature_df.to_csv(Constant.EXTENDED_TOP_DATASET_PATH + '/training_attack_feature.csv',
                                              index=False)
            testing_normal_feature_df.to_csv(Constant.EXTENDED_TOP_DATASET_PATH + '/testing_normal_feature.csv',
                                             index=False)
            testing_attack_feature_df.to_csv(Constant.EXTENDED_TOP_DATASET_PATH + '/testing_attack_feature.csv',
                                             index=False)
        else:
            training_normal_feature_df = pd.DataFrame(training_normal_feature_dict)
            training_attack_feature_df = pd.DataFrame(training_attack_feature_dict)
            testing_normal_feature_df = pd.DataFrame(testing_normal_feature_dict)
            testing_attack_feature_df = pd.DataFrame(testing_attack_feature_dict)

            training_normal_feature_df.to_csv(Constant.CUT_TOP_DATASET_PATH + '/training_normal_feature.csv',
                                              index=False)
            training_attack_feature_df.to_csv(Constant.CUT_TOP_DATASET_PATH + '/training_attack_feature.csv',
                                              index=False)
            testing_normal_feature_df.to_csv(Constant.CUT_TOP_DATASET_PATH + '/testing_normal_feature.csv',
                                             index=False)
            testing_attack_feature_df.to_csv(Constant.CUT_TOP_DATASET_PATH + '/testing_attack_feature.csv',
                                             index=False)

    @classmethod
    def save_stat_features(cls, training_feature_array, training_label_array, testing_feature_array,
                           testing_label_array, feature_type):
        df = pd.DataFrame(training_feature_array)
        df.to_csv(Constant.SINGLE_STAT_DATASET_PATH + '/training_combined_' + feature_type + '_feature.csv',
                  index=False)
        df = pd.DataFrame(training_label_array)
        df.to_csv(Constant.SINGLE_STAT_DATASET_PATH + '/training_combined_' + feature_type + '_label.csv', index=False)

        df = pd.DataFrame(testing_feature_array)
        df.to_csv(Constant.SINGLE_STAT_DATASET_PATH + '/testing_combined_' + feature_type + '_feature.csv',
                  index=False)
        df = pd.DataFrame(testing_label_array)
        df.to_csv(Constant.SINGLE_STAT_DATASET_PATH + '/testing_combined_' + feature_type + '_label.csv', index=False)

