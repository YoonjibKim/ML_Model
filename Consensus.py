import csv
import os
import numpy as np
import Constant
from Machine_learning_algorithms.DNN import DNN
from Machine_learning_algorithms.Gaussian_NB import Gaussian_NB
from Machine_learning_algorithms.K_Means import K_Means
from Machine_learning_algorithms.K_Nearest_Neighbor import KNN
from Machine_learning_algorithms.Linear_Regressions import Linear_Regressions
from Machine_learning_algorithms.Logistic_Regression import Logistic_Regression


class Consensus(KNN, K_Means, DNN, Logistic_Regression, Gaussian_NB, Linear_Regressions):
    __training_normal_data_array = None
    __training_attack_data_array = None
    __testing_normal_data_array = None
    __testing_attack_data_array = None
    __training_normal_label_array = None
    __training_attack_label_array = None
    __testing_normal_label_array = None
    __testing_attack_label_array = None
    __training_data_array = None
    __training_label_array = None
    __testing_data_array = None
    __testing_label_array = None

    def __init__(self, data_path):
        self.__load_data(data_path)

    @classmethod
    def __load_data(cls, root_path):
        file_name_list = []
        _file_name_list = os.listdir(root_path)
        for file_name in _file_name_list:
            if file_name.find('.csv') > 0:
                file_name_list.append(file_name)

        data_path_list = []
        for file_name in file_name_list:
            data_path_list.append(root_path + '/' + file_name)

        dim_flag = True
        for data_path in data_path_list:
            read_line = []
            with open(data_path, 'r') as f:
                rdr = csv.reader(f)
                for line in rdr:
                    read_line.append(line)

            if dim_flag:
                dim_for_data = len(read_line[0])
                cls.__training_normal_data_array = np.empty((0, dim_for_data), float)
                cls.__training_attack_data_array = np.empty((0, dim_for_data), float)
                cls.__testing_normal_data_array = np.empty((0, dim_for_data), float)
                cls.__testing_attack_data_array = np.empty((0, dim_for_data), float)
                dim_flag = False

            split_path_list = data_path.split('/')
            file_name = split_path_list[len(split_path_list) - 1]
            feature_array = np.array(read_line[1:])
            if file_name == Constant.TESTING_ATTACK:
                cls.__testing_attack_data_array = np.append(cls.__testing_attack_data_array, feature_array, axis=0)
                temp_list = []
                for i in range(0, len(cls.__testing_attack_data_array)):
                    temp_list.append(1)
                cls.__testing_attack_label_array = np.array(temp_list)
            elif file_name == Constant.TESTING_NORMAL:
                cls.__testing_normal_data_array = np.append(cls.__testing_normal_data_array, feature_array, axis=0)
                temp_list = []
                for i in range(0, len(cls.__testing_normal_data_array)):
                    temp_list.append(0)
                cls.__testing_normal_label_array = np.array(temp_list)
            elif file_name == Constant.TRAINING_ATTACK:
                cls.__training_attack_data_array = np.append(cls.__training_attack_data_array, feature_array, axis=0)
                temp_list = []
                for i in range(0, len(cls.__training_attack_data_array)):
                    temp_list.append(1)
                cls.__training_attack_label_array = np.array(temp_list)
            elif file_name == Constant.TRAINING_NORMAL:
                cls.__training_normal_data_array = np.append(cls.__training_normal_data_array, feature_array, axis=0)
                temp_list = []
                for i in range(0, len(cls.__training_normal_data_array)):
                    temp_list.append(0)
                cls.__training_normal_label_array = np.array(temp_list)

        cls.__set_ml_features()

    @classmethod
    def __set_ml_features(cls):
        training_data_array = np.append(cls.__training_normal_data_array, cls.__training_attack_data_array,
                                        axis=0)
        cls.__training_data_array = np.asarray(training_data_array, dtype=float)
        training_label_array = np.append(cls.__training_normal_label_array, cls.__training_attack_label_array,
                                         axis=0)
        cls.__training_label_array = np.asarray(training_label_array, dtype=float)
        testing_data_array = np.append(cls.__testing_normal_data_array, cls.__testing_attack_data_array,
                                       axis=0)
        cls.__testing_data_array = np.asarray(testing_data_array, dtype=float)
        testing_label_array = np.append(cls.__testing_normal_label_array, cls.__testing_attack_label_array,
                                        axis=0)
        cls.__testing_label_array = np.asarray(testing_label_array, dtype=float)

    @classmethod
    def get_ml_training_and_testing_features(cls):
        return cls.__training_data_array, cls.__testing_data_array, cls.__training_label_array, \
            cls.__testing_label_array

    @classmethod
    def __add_label_to_feature_list(cls, feature_list, label):
        temp_feature_list = list(feature_list)
        combined_feature_list = []
        for record in temp_feature_list:
            temp_record = []
            temp_record.extend(record)
            temp_record.append(label)
            combined_feature_list.append(temp_record)

        return combined_feature_list

    @classmethod
    def divide_attack_and_normal_features(cls):
        training_normal_feature_list = \
            cls.__add_label_to_feature_list(cls.__training_normal_data_array, str(Constant.NORMAL_LABEL))
        training_attack_feature_list = \
            cls.__add_label_to_feature_list(cls.__training_attack_data_array, str(Constant.ATTACK_LABEL))
        testing_normal_feature_list = \
            cls.__add_label_to_feature_list(cls.__testing_normal_data_array, str(Constant.NORMAL_LABEL))
        testing_attack_feature_list = \
            cls.__add_label_to_feature_list(cls.__testing_attack_data_array, str(Constant.ATTACK_LABEL))

        attack_feature_list = []
        attack_feature_list.extend(training_attack_feature_list)
        attack_feature_list.extend(testing_attack_feature_list)

        normal_feature_list = []
        normal_feature_list.extend(training_normal_feature_list)
        normal_feature_list.extend(testing_normal_feature_list)

        return attack_feature_list, normal_feature_list

    @classmethod
    def __convert_training_and_testing_features(cls, training_feature_array, training_label_array,
                                                testing_feature_array, testing_label_array):
        X_tn = training_feature_array
        X_te = testing_feature_array
        y_tn = training_label_array
        y_te = testing_label_array

        return X_tn, y_tn, X_te, y_te

    @classmethod
    def __convert_testing_feature(cls, testing_feature_array, testing_label_array):
        X_te = testing_feature_array
        y_te = testing_label_array

        return X_te, y_te

    @classmethod
    def knn(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        super().knn_run(training_feature_array, training_label_array, testing_feature_array, testing_label_array)

    @classmethod
    def k_means(cls, testing_feature_array, testing_label_array):
        X_te, y_te = cls.__convert_testing_feature(testing_feature_array, testing_label_array)
        super().k_means_run(testing_feature_array, testing_label_array)

    @classmethod
    def dnn(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        super().dnn_run(training_feature_array, training_label_array, testing_feature_array, testing_label_array)

    @classmethod
    def logistic_regression(cls, training_feature_array, training_label_array, testing_feature_array,
                            testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        super().logistic_regression_run(training_feature_array, training_label_array, testing_feature_array,
                                        testing_label_array)

    @classmethod
    def gaussian_nb(cls, training_feature_array, training_label_array, testing_feature_array,
                    testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        super().gaussian_nb_run(training_feature_array, training_label_array, testing_feature_array,
                                testing_label_array)

    @classmethod
    def linear_regressions(cls, training_feature_array, training_label_array, testing_feature_array,
                           testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        super().linear_regressions_run(X_tn, y_tn, X_te, y_te)
