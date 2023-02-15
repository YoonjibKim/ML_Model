import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import Constant
from Machine_learning_algorithms.K_Nearest_Neighbor import KNN


class Consensus(KNN):
    __training_normal_feature_dict = {}
    __training_attack_feature_dict = {}
    __testing_normal_feature_dict = {}
    __testing_attack_feature_dict = {}

    def __init__(self, training_normal_data_dict, training_attack_data_dict, testing_normal_data, testing_attack_data):
        self.__training_normal_feature_dict, self.__training_attack_feature_dict, self.__testing_normal_feature_dict, \
            self.__testing_attack_feature_dict = self.__prepare_features(training_normal_data_dict,
                                                                         training_attack_data_dict, testing_normal_data,
                                                                         testing_attack_data)
        self.__training_normal_data_dict = training_normal_data_dict
        self.__training_attack_data_dict = training_attack_data_dict
        self.__testing_normal_data = testing_normal_data
        self.__testing_normal_data = testing_attack_data

    def get_extended_features(self):
        return self.__training_normal_feature_dict, self.__training_attack_feature_dict, \
            self.__testing_normal_feature_dict, self.__testing_attack_feature_dict

    @classmethod
    def __get_extended_data_points(cls, symbol, feature_array, extension_size, feature_type):
        extended_y_array = None
        x_list = []
        y = np.float_(list(map(lambda data: [data], feature_array)))
        for index in range(0, len(feature_array)):
            x_list.append([index])

        x = np.float_(x_list)
        extension_size = int(extension_size)
        if len(feature_array) < extension_size - 1:
            extension_size -= len(feature_array)
            output = None

            final_x = x_list[len(x_list) - 1][0]
            x = torch.Tensor(x)
            y = torch.Tensor(y)

            model = nn.Sequential(
                nn.Linear(1, 5),
                nn.LeakyReLU(0.2),
                nn.Linear(5, 10),
                nn.LeakyReLU(0.2),
                nn.Linear(10, 10),
                nn.LeakyReLU(0.2),
                nn.Linear(10, 10),
                nn.LeakyReLU(0.2),
                nn.Linear(10, 5),
                nn.LeakyReLU(0.2),
                nn.Linear(5, 1),
            )

            cpu = torch.device('cpu')
            loss_func = nn.L1Loss().to(cpu)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
            model = model.to(cpu)

            x = x.to(cpu)
            y = y.to(cpu)

            added_x = torch.Tensor(list([index] for index in range(final_x + 1, final_x + extension_size)))

            num_epoch = Constant.FEATURE_EPOCH
            loss_array = []
            for epoch in range(num_epoch):
                optimizer.zero_grad()
                output = model(x)

                loss = loss_func(output, y)
                loss.backward()
                optimizer.step()

                loss_array.append(loss)
                if epoch % 100 == 0:
                    print('epoch:', epoch, ' loss:', loss.item())

            added_output = model(added_x)

            x = x.cpu().detach().numpy()
            y = y.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            added_output = added_output.cpu().detach().numpy()
            extended_y_array = np.concatenate((output, added_output))

            x_list = []
            for _x in range(len(x) + 1, len(x) + extension_size):
                x_list.append([_x])

            x_array = np.array(x_list)
            extended_x_array = np.concatenate((x, x_array))
            extended_x_array = torch.Tensor(extended_x_array)
        else:
            if len(feature_array) > extension_size:
                feature_type_size = len(feature_array)
                term = feature_type_size / extension_size
                term_round = round(term)

                data_list = []
                x_list.clear()
                x_index = 0
                feature_size = len(feature_array)
                for index in range(0, feature_size):
                    if x_index >= extension_size - 1:
                        break

                    residue = index % term_round
                    if residue == 0:
                        term_data = feature_array[index]
                        data_list.append(term_data)
                        x_list.append(x_index)
                        x_index += 1

                x = np.array(x_list)
                feature_array = np.array(data_list)
            else:
                while len(feature_array) != extension_size:
                    last_one = feature_array[len(feature_array) - 1]
                    feature_array = np.append(feature_array, [last_one], axis=0)
                    last_count = x[len(x) - 1]
                    last_count += 1.0
                    x = np.append(x, [last_count], axis=0)

            print(feature_type, len(feature_array), extension_size)
            extended_y_array = np.array(feature_array)

        return extended_y_array

    def __prepare_features(self, training_normal_data_dict, training_attack_data_dict, testing_normal_data_dict,
                           testing_attack_data_dict):
        training_data_point_size_list = []
        for symbol, data_list in training_normal_data_dict.items():
            size = len(data_list)
            training_data_point_size_list.append(size)

        for symbol, data_list in training_attack_data_dict.items():
            size = len(data_list)
            training_data_point_size_list.append(size)

        testing_data_point_size_list = []
        for symbol, data_list in testing_normal_data_dict.items():
            size = len(data_list)
            testing_data_point_size_list.append(size)

        for symbol, data_list in testing_attack_data_dict.items():
            size = len(data_list)
            testing_data_point_size_list.append(size)

        # 하드에 저장
        training_mean = np.mean(training_data_point_size_list)
        training_std = np.std(training_data_point_size_list)
        testing_mean = np.mean(testing_data_point_size_list)
        testing_std = np.mean(testing_data_point_size_list)

        sorted_training_list = sorted(training_data_point_size_list, reverse=True)
        max_training_size = sorted_training_list[0]
        fixed_testing_size = (max_training_size * 0.25) / 0.75

        training_normal_feature_dict = {}
        for symbol, feature_list in training_normal_data_dict.items():
            print('training normal', len(feature_list), max_training_size)  # 파일에 저장
            temp = self.__get_extended_data_points(symbol, feature_list, max_training_size,
                                                   Constant.TRAINING_NORMAL)
            training_normal_feature_dict[symbol] = temp

        training_attack_feature_dict = {}
        for symbol, feature_list in training_attack_data_dict.items():
            print('training attack', len(feature_list), max_training_size)  # 파일에 저장
            temp = self.__get_extended_data_points(symbol, feature_list, max_training_size,
                                                   Constant.TRAINING_ATTACK)
            training_attack_feature_dict[symbol] = temp

        size_list = []
        temp_testing_normal_feature_dict = {}
        for symbol, feature_list in testing_normal_data_dict.items():
            print('testing normal', len(feature_list), fixed_testing_size)  # 파일에 저장
            temp = self.__get_extended_data_points(symbol, feature_list, fixed_testing_size,
                                                   Constant.TESTING_NORMAL)
            temp_testing_normal_feature_dict[symbol] = temp
            size_list.append(len(temp))

        size_list = sorted(size_list)
        testing_normal_min_size = size_list[0]
        size_list.clear()

        temp_testing_attack_feature_dict = {}
        for symbol, feature_list in testing_attack_data_dict.items():
            print('testing attack', len(feature_list), fixed_testing_size)  # 파일에 저장
            temp = self.__get_extended_data_points(symbol, feature_list, fixed_testing_size,
                                                   Constant.TESTING_ATTACK)
            temp_testing_attack_feature_dict[symbol] = temp
            size_list.append(len(temp))

        size_list = sorted(size_list)
        testing_attack_min_size = size_list[0]

        if testing_normal_min_size < testing_attack_min_size:
            testing_min_size = testing_normal_min_size
        else:
            testing_min_size = testing_attack_min_size

        testing_normal_feature_dict = {}
        for symbol, data_points in temp_testing_normal_feature_dict.items():
            testing_normal_feature_dict[symbol] = data_points[:testing_min_size]

        testing_attack_feature_dict = {}
        for symbol, data_points in temp_testing_attack_feature_dict.items():
            testing_attack_feature_dict[symbol] = data_points[:testing_min_size]

        return training_normal_feature_dict, training_attack_feature_dict, testing_normal_feature_dict, \
            testing_attack_feature_dict

    def knn(self):
        KNN.run(self, self.__training_normal_data_dict, self.__training_attack_data_dict, self.__testing_normal_data,
                self.__testing_normal_data)

