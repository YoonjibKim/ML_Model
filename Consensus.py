
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

import Constant
from Machine_learning_algorithms.K_Nearest_Neighbor import KNN


class Consensus(KNN):
    def __init__(self, training_normal_data_dict, training_attack_data_dict, testing_normal_data, testing_attack_data):
        self.__prepare_features(training_normal_data_dict, training_attack_data_dict, testing_normal_data,
                                testing_attack_data)
        self.__training_normal_data_dict = training_normal_data_dict
        self.__training_attack_data_dict = training_attack_data_dict
        self.__testing_normal_data = testing_normal_data
        self.__testing_normal_data = testing_attack_data

    @classmethod
    def __get_extended_data_points(cls, symbol, feature_list, extension_size, feature_type):
        extended_y_array = None
        if len(feature_list) < extension_size:
            extension_size -= len(feature_list)
            output = None
            x_list = []
            y = np.float_(list(map(lambda data: [data], feature_list)))
            for index in range(0, len(feature_list)):
                x_list.append([index])

            final_x = x_list[len(x_list) - 1][0]

            x = np.float_(x_list)
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

            num_epoch = 200
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

            fi_los = [fl.item() for fl in loss_array]
            plt.plot(range(num_epoch), fi_los)
            plt.title('Adam Optimizer: ' + symbol)
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.savefig('Output_results/' + feature_type + '/loss_rate_' + symbol + '.png')

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

            plt.figure(figsize=(10, 10))
            plt.title('Non-linear Regression Analysis: ' + symbol)
            plt.ylabel('overhead (%)')
            plt.xlabel('sequence')
            plt.plot(x, y, '-', color="blue", label='Solid')
            plt.plot(extended_x_array, extended_y_array, '-', color="red", label='Solid')
            plt.savefig('Output_results/' + feature_type + '/objective_function_' + symbol + '.png')
        else:
            extended_y_array = np.array(feature_list)

        return extended_y_array

    def __prepare_features(self, training_normal_data_dict, training_attack_data_dict, testing_normal_data,
                           testing_attack_data):
        training_data_point_size_list = []
        for symbol, data_list in training_normal_data_dict.items():
            size = len(data_list)
            training_data_point_size_list.append(size)

        for symbol, data_list in training_attack_data_dict.items():
            size = len(data_list)
            training_data_point_size_list.append(size)

        testing_data_point_size_list = []
        for symbol, data_list in testing_normal_data.items():
            size = len(data_list)
            testing_data_point_size_list.append(size)

        for symbol, data_list in testing_attack_data.items():
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
            print(len(feature_list), max_training_size)
            temp = self.__get_extended_data_points(symbol, feature_list, max_training_size,
                                                   Constant.TRAINING_NORMAL)
            training_normal_feature_dict[symbol] = temp

        training_attack_feature_dict = {}
        for symbol, feature_list in training_attack_data_dict.items():
            print(len(feature_list), max_training_size)
            temp = self.__get_extended_data_points(symbol, feature_list, max_training_size,
                                                   Constant.TRAINING_ATTACK)
            training_attack_feature_dict[symbol] = temp



    def knn(self):
        KNN.run(self, self.__training_normal_data_dict, self.__training_attack_data_dict, self.__testing_normal_data,
                self.__testing_normal_data)

