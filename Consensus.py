import csv
import os
import numpy as np
import Constant
from APRF import APRF
from Machine_learning_algorithms.Ada_Boost import Ada_Boost
from Machine_learning_algorithms.Agglomerative_Clustering import Agglomerative_Clustering
from Machine_learning_algorithms.DB_Scan import DB_Scan
from Machine_learning_algorithms.DNN import DNN
from Machine_learning_algorithms.Decision_Tree import Decision_Tree
from Machine_learning_algorithms.Gaussian_Mixture import Gaussian_Mixture
from Machine_learning_algorithms.Gaussian_NB import Gaussian_NB
from Machine_learning_algorithms.Gradient_Boost import Gradient_Boost
from Machine_learning_algorithms.K_Means import K_Means
from Machine_learning_algorithms.KNN import KNN
from Machine_learning_algorithms.Linear_Regressions import Linear_Regressions
from Machine_learning_algorithms.Logistic_Regression import Logistic_Regression
from Machine_learning_algorithms.Random_Forest import Random_Forest
from Machine_learning_algorithms.SVM import SVM


class Consensus(KNN, K_Means, DNN, Logistic_Regression, Gaussian_NB, Linear_Regressions, Decision_Tree, SVM,
                Random_Forest, Ada_Boost, Gradient_Boost, DB_Scan, Gaussian_Mixture, Agglomerative_Clustering, APRF):
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

    __knn_score_list = []
    __knn_candidate_score_list = []
    __k_means_score_list = []
    __k_means_candidate_score_list = []
    __dnn_score_list = []
    __dnn_candidate_score_list = []
    __logistic_regression_score_list = []
    __logistic_regression_candidate_score_list = []
    __gaussian_nb_score_list = []
    __gaussian_nb_candidate_score_list = []
    __linear_regression_score_list = []
    __linear_regression_candidate_score_list = []
    __decision_tree_score_list = []
    __decision_tree_candidate_score_list = []
    __svm_score_list = []
    __svm_score_candidate_score_list = []
    __random_forest_score_list = []
    __random_forest_candidate_score_list = []
    __ada_boost_score_list = []
    __ada_boost_candidate_score_list = []
    __gradient_boost_score_list = []
    __gradient_boost_candidate_score_list = []
    __db_scan_score_list = []
    __db_scan_candidate_score_list = []
    __gaussian_mixture_score_list = []
    __gaussian_mixture_candidate_score_list = []
    __agglomerative_clustering_score_list = []
    __agglomerative_clustering_candidate_score_list = []
    __linear_regression_elastic_score_list = []
    __linear_regression_elastic_candidate_score_list = []
    __linear_regression_lasso_score_list = []
    __linear_regression_lasso_candidate_score_list = []
    __linear_regression_ridge_score_list = []
    __linear_regression_ridge_candidate_score_list = []

    __ada_boost_negative_score = 0
    __ada_boost_positive_score = 0
    __agglomerative_clustering_negative_score = 0
    __agglomerative_clustering_positive_score = 0
    __db_scan_negative_score = 0
    __db_scan_positive_score = 0
    __decision_tree_negative_score = 0
    __decision_tree_positive_score = 0
    __dnn_negative_score = 0
    __dnn_positive_score = 0
    __gaussian_mixture_negative_score = 0
    __gaussian_mixture_positive_score = 0
    __gaussian_nb_negative_score = 0
    __gaussian_nb_positive_score = 0
    __gradient_boost_negative_score = 0
    __gradient_boost_positive_score = 0
    __k_means_negative_score = 0
    __k_means_positive_score = 0
    __knn_negative_score = 0
    __knn_positive_score = 0
    __linear_regression_negative_score = 0
    __linear_regression_positive_score = 0
    __linear_regression_ridge_negative_score = 0
    __linear_regression_ridge_positive_score = 0
    __linear_regression_lasso_negative_score = 0
    __linear_regression_lasso_positive_score = 0
    __linear_regression_elastic_negative_score = 0
    __linear_regression_elastic_positive_score = 0
    __logistic_regression_negative_score = 0
    __logistic_regression_positive_score = 0
    __random_forest_negative_score = 0
    __random_forest_positive_score = 0
    __svm_negative_score = 0
    __svm_positive_score = 0

    def __init__(self, data_path):
        self.__load_data(data_path)

    @classmethod
    def __init_all_variables(cls):
        cls.__knn_score_list.clear()
        cls.__knn_candidate_score_list.clear()
        cls.__k_means_score_list.clear()
        cls.__k_means_candidate_score_list.clear()
        cls.__dnn_score_list.clear()
        cls.__dnn_candidate_score_list.clear()
        cls.__logistic_regression_score_list.clear()
        cls.__logistic_regression_candidate_score_list.clear()
        cls.__gaussian_nb_score_list.clear()
        cls.__gaussian_nb_candidate_score_list.clear()
        cls.__linear_regression_score_list.clear()
        cls.__linear_regression_candidate_score_list.clear()
        cls.__decision_tree_score_list.clear()
        cls.__decision_tree_candidate_score_list.clear()
        cls.__svm_score_list.clear()
        cls.__svm_score_candidate_score_list.clear()
        cls.__random_forest_score_list.clear()
        cls.__random_forest_candidate_score_list.clear()
        cls.__ada_boost_score_list.clear()
        cls.__ada_boost_candidate_score_list.clear()
        cls.__gradient_boost_score_list.clear()
        cls.__gradient_boost_candidate_score_list.clear()
        cls.__db_scan_score_list.clear()
        cls.__db_scan_candidate_score_list.clear()
        cls.__gaussian_mixture_score_list.clear()
        cls.__gaussian_mixture_candidate_score_list.clear()
        cls.__agglomerative_clustering_score_list.clear()
        cls.__agglomerative_clustering_candidate_score_list.clear()
        cls.__linear_regression_elastic_score_list.clear()
        cls.__linear_regression_elastic_candidate_score_list.clear()
        cls.__linear_regression_lasso_score_list.clear()
        cls.__linear_regression_lasso_candidate_score_list.clear()
        cls.__linear_regression_ridge_score_list.clear()
        cls.__linear_regression_ridge_candidate_score_list.clear()

        cls.__ada_boost_negative_score = 0
        cls.__ada_boost_positive_score = 0
        cls.__agglomerative_clustering_negative_score = 0
        cls.__agglomerative_clustering_positive_score = 0
        cls.__db_scan_negative_score = 0
        cls.__db_scan_positive_score = 0
        cls.__decision_tree_negative_score = 0
        cls.__decision_tree_positive_score = 0
        cls.__dnn_negative_score = 0
        cls.__dnn_positive_score = 0
        cls.__gaussian_mixture_negative_score = 0
        cls.__gaussian_mixture_positive_score = 0
        cls.__gaussian_nb_negative_score = 0
        cls.__gaussian_nb_positive_score = 0
        cls.__gradient_boost_negative_score = 0
        cls.__gradient_boost_positive_score = 0
        cls.__k_means_negative_score = 0
        cls.__k_means_positive_score = 0
        cls.__knn_negative_score = 0
        cls.__knn_positive_score = 0
        cls.__linear_regression_negative_score = 0
        cls.__linear_regression_positive_score = 0
        cls.__linear_regression_ridge_negative_score = 0
        cls.__linear_regression_ridge_positive_score = 0
        cls.__linear_regression_lasso_negative_score = 0
        cls.__linear_regression_lasso_positive_score = 0
        cls.__linear_regression_elastic_negative_score = 0
        cls.__linear_regression_elastic_positive_score = 0
        cls.__logistic_regression_negative_score = 0
        cls.__logistic_regression_positive_score = 0
        cls.__random_forest_negative_score = 0
        cls.__random_forest_positive_score = 0
        cls.__svm_negative_score = 0
        cls.__svm_positive_score = 0

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
        X = testing_feature_array
        y = testing_label_array

        return X, y

    @classmethod
    def __knn(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        return super().knn_run(X_tn, y_tn, X_te, y_te)

    @classmethod
    def __k_means(cls, testing_feature_array, testing_label_array):
        X, y = cls.__convert_testing_feature(testing_feature_array, testing_label_array)
        return super().k_means_run(X, y)

    @classmethod
    def __dnn(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        return super().dnn_run(X_tn, y_tn, X_te, y_te)

    @classmethod
    def __logistic_regression(cls, training_feature_array, training_label_array, testing_feature_array,
                              testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        return super().logistic_regression_run(X_tn, y_tn, X_te, y_te)

    @classmethod
    def __gaussian_nb(cls, training_feature_array, training_label_array, testing_feature_array,
                      testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        return super().gaussian_nb_run(X_tn, y_tn, X_te, y_te)

    @classmethod
    def __linear_regressions(cls, training_feature_array, training_label_array, testing_feature_array,
                             testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        return super().linear_regressions_run(X_tn, y_tn, X_te, y_te)

    @classmethod
    def __decision_tree(cls, training_feature_array, training_label_array, testing_feature_array,
                        testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        return super().decision_tree_run(X_tn, y_tn, X_te, y_te)

    @classmethod
    def __svm(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        return super().svm_run(X_tn, y_tn, X_te, y_te)

    @classmethod
    def __random_forest(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        return super().random_forest_run(X_tn, y_tn, X_te, y_te)

    @classmethod
    def __ada_boost(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        return super().ada_boost_run(X_tn, y_tn, X_te, y_te)

    @classmethod
    def __gradient_boost(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        X_tn, y_tn, X_te, y_te = cls.__convert_training_and_testing_features(training_feature_array,
                                                                             training_label_array,
                                                                             testing_feature_array,
                                                                             testing_label_array)
        return super().gradient_boost_run(X_tn, y_tn, X_te, y_te)

    @classmethod
    def __db_scan(cls, testing_feature_array, testing_label_array):
        X, y = cls.__convert_testing_feature(testing_feature_array, testing_label_array)
        return super().db_scan_run(X, y)

    @classmethod
    def __gaussian_mixture(cls, testing_feature_array, testing_label_array):
        X, y = cls.__convert_testing_feature(testing_feature_array, testing_label_array)
        return super().gaussian_mixture_run(X, y)

    @classmethod
    def __agglomerative_clustering(cls, testing_feature_array, testing_label_array):
        X, y = cls.__convert_testing_feature(testing_feature_array, testing_label_array)
        return super().agglomerative_clustering_run(X, y)

    @classmethod
    def __calculate_scores(cls, class_report_list):
        for index, outer_list in enumerate(class_report_list):
            for class_report in outer_list:
                try:
                    normal_count = class_report[str(Constant.NORMAL_LABEL)][Constant.SUPPORT]
                except KeyError:
                    normal_count = class_report[str(Constant.NORMAL_LABEL_FLOAT)][Constant.SUPPORT]
                try:
                    attack_count = class_report[str(Constant.ATTACK_LABEL)][Constant.SUPPORT]
                except KeyError:
                    attack_count = class_report[str(Constant.ATTACK_LABEL_FLOAT)][Constant.SUPPORT]

                f1_score = class_report[str(Constant.WEIGHTED_AVG)][Constant.F1_SCORE]

                total_count = normal_count + attack_count
                positive_ratio = normal_count / total_count
                negative_ratio = attack_count / total_count
                positive_score = f1_score * positive_ratio
                negative_score = (1.0 - f1_score) * negative_ratio

                if Constant.ADA_BOOST in class_report.keys():
                    print('ada boost')
                    cls.__ada_boost_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__ada_boost_positive_score += positive_score
                    cls.__ada_boost_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__ada_boost_positive_score, cls.__ada_boost_negative_score)
                    cls.__ada_boost_candidate_score_list.append(candidate_score)
                elif Constant.AGGLOMERATIVE_CLUSTERING in class_report.keys():
                    print('agglomerative clustering')
                    cls.__agglomerative_clustering_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__agglomerative_clustering_positive_score += positive_score
                    cls.__agglomerative_clustering_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__agglomerative_clustering_positive_score,
                                                       cls.__agglomerative_clustering_negative_score)
                    cls.__agglomerative_clustering_candidate_score_list.append(candidate_score)
                elif Constant.DB_SCAN in class_report.keys():
                    print('db scan')
                    cls.__db_scan_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__db_scan_positive_score += positive_score
                    cls.__db_scan_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__db_scan_positive_score, cls.__db_scan_negative_score)
                    cls.__db_scan_candidate_score_list.append(candidate_score)
                elif Constant.DECISION_TREE in class_report.keys():
                    print('decision tree')
                    cls.__decision_tree_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__decision_tree_positive_score += positive_score
                    cls.__decision_tree_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__decision_tree_positive_score,
                                                       cls.__decision_tree_negative_score)
                    cls.__decision_tree_candidate_score_list.append(candidate_score)
                elif Constant.DNN in class_report.keys():
                    print('dnn')
                    cls.__dnn_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__dnn_positive_score += positive_score
                    cls.__dnn_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__dnn_positive_score, cls.__dnn_negative_score)
                    cls.__dnn_candidate_score_list.append(candidate_score)
                elif Constant.GAUSSIAN_MIXTURE in class_report.keys():
                    print('gaussian mixture')
                    cls.__gaussian_mixture_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__gaussian_mixture_positive_score += positive_score
                    cls.__gaussian_mixture_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__gaussian_mixture_positive_score,
                                                       cls.__gaussian_mixture_negative_score)
                    cls.__gaussian_mixture_candidate_score_list.append(candidate_score)
                elif Constant.GAUSSIAN_NB in class_report.keys():
                    print('gaussian nb')
                    cls.__gaussian_nb_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__gaussian_nb_positive_score += positive_score
                    cls.__gaussian_nb_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__gaussian_nb_positive_score,
                                                       cls.__gaussian_nb_negative_score)
                    cls.__gaussian_nb_candidate_score_list.append(candidate_score)
                elif Constant.GRADIENT_BOOST in class_report.keys():
                    print('gradient boost')
                    cls.__gradient_boost_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__gradient_boost_positive_score += positive_score
                    cls.__gradient_boost_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__gradient_boost_positive_score,
                                                       cls.__gradient_boost_negative_score)
                    cls.__gradient_boost_candidate_score_list.append(candidate_score)
                elif Constant.K_MEANS in class_report.keys():
                    print('k means')
                    cls.__k_means_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__k_means_positive_score += positive_score
                    cls.__k_means_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__k_means_positive_score, cls.__k_means_negative_score)
                    cls.__k_means_candidate_score_list.append(candidate_score)
                elif Constant.KNN in class_report.keys():
                    print('knn')
                    cls.__knn_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__knn_positive_score += positive_score
                    cls.__knn_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__knn_positive_score, cls.__knn_negative_score)
                    cls.__knn_candidate_score_list.append(candidate_score)
                elif Constant.LOGISTIC_REGRESSION in class_report.keys():
                    print('logistic regression')
                    cls.__logistic_regression_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__logistic_regression_positive_score += positive_score
                    cls.__logistic_regression_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__logistic_regression_positive_score,
                                                       cls.__logistic_regression_negative_score)
                    cls.__logistic_regression_candidate_score_list.append(candidate_score)
                elif Constant.RANDOM_FOREST in class_report.keys():
                    print('random forest')
                    cls.__random_forest_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__random_forest_positive_score += positive_score
                    cls.__random_forest_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__random_forest_positive_score,
                                                       cls.__random_forest_negative_score)
                    cls.__random_forest_candidate_score_list.append(candidate_score)
                elif Constant.SVM in class_report.keys():
                    print('svm')
                    cls.__svm_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__svm_positive_score += positive_score
                    cls.__svm_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__svm_positive_score, cls.__svm_negative_score)
                    cls.__svm_score_candidate_score_list.append(candidate_score)
                elif Constant.LINEAR_REGRESSION in class_report.keys():
                    print('linear regression')
                    cls.__linear_regression_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__linear_regression_positive_score += positive_score
                    cls.__linear_regression_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__linear_regression_positive_score,
                                                       cls.__linear_regression_negative_score)
                    cls.__linear_regression_candidate_score_list.append(candidate_score)
                elif Constant.LINEAR_REGRESSION_ELASTIC in class_report.keys():
                    print('linear regression elastic')
                    cls.__linear_regression_ridge_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__linear_regression_ridge_positive_score += positive_score
                    cls.__linear_regression_ridge_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__linear_regression_ridge_positive_score,
                                                       cls.__linear_regression_ridge_negative_score)
                    cls.__linear_regression_ridge_candidate_score_list.append(candidate_score)
                elif Constant.LINEAR_REGRESSION_LASSO in class_report.keys():
                    print('linear regressio lasso')
                    cls.__linear_regression_lasso_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__linear_regression_lasso_positive_score += positive_score
                    cls.__linear_regression_lasso_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__linear_regression_lasso_positive_score,
                                                       cls.__linear_regression_lasso_negative_score)
                    cls.__linear_regression_lasso_candidate_score_list.append(candidate_score)
                elif Constant.LINEAR_REGRESSION_RIDGE in class_report.keys():
                    print('linear regressio ridge')
                    cls.__linear_regression_ridge_score_list.append(class_report[Constant.WEIGHTED_AVG])
                    cls.__linear_regression_ridge_positive_score += positive_score
                    cls.__linear_regression_ridge_negative_score += negative_score
                    candidate_score = super().aprf_run(cls.__linear_regression_ridge_positive_score,
                                                       cls.__linear_regression_ridge_negative_score)
                    cls.__linear_regression_ridge_candidate_score_list.append(candidate_score)
                else:
                    print('wrong ML')

    @classmethod
    def __get_best_ml(cls, training_feature_array, training_label_array, testing_feature_array,
                      testing_label_array):
        class_report_list = []

        class_report = \
            cls.__knn(training_feature_array, training_label_array, testing_feature_array, testing_label_array)
        class_report[Constant.KNN] = Constant.KNN
        class_report_list.append([class_report])

        class_report = \
            cls.__k_means(testing_feature_array, testing_label_array)
        class_report[Constant.K_MEANS] = Constant.K_MEANS
        class_report_list.append([class_report])

        class_report = \
            cls.__dnn(training_feature_array, training_label_array, testing_feature_array, testing_label_array)
        class_report[Constant.DNN] = Constant.DNN
        class_report_list.append([class_report])

        class_report = \
            cls.__logistic_regression(training_feature_array, training_label_array, testing_feature_array,
                                      testing_label_array)
        class_report[Constant.LOGISTIC_REGRESSION] = Constant.LOGISTIC_REGRESSION
        class_report_list.append([class_report])

        class_report = \
            cls.__gaussian_nb(training_feature_array, training_label_array, testing_feature_array, testing_label_array)
        class_report[Constant.GAUSSIAN_NB] = Constant.GAUSSIAN_NB
        class_report_list.append([class_report])

        class_report_lr, class_report_ridge, class_report_lasso, class_report_elastic = \
            cls.__linear_regressions(training_feature_array, training_label_array, testing_feature_array,
                                     testing_label_array)
        class_report_lr[Constant.LINEAR_REGRESSION] = Constant.LINEAR_REGRESSION
        class_report_ridge[Constant.LINEAR_REGRESSION_RIDGE] = Constant.LINEAR_REGRESSION_RIDGE
        class_report_lasso[Constant.LINEAR_REGRESSION_LASSO] = Constant.LINEAR_REGRESSION_LASSO
        class_report_elastic[Constant.LINEAR_REGRESSION_ELASTIC] = Constant.LINEAR_REGRESSION_ELASTIC
        class_report_list.append([class_report_lr, class_report_ridge, class_report_lasso, class_report_elastic])

        class_report = \
            cls.__decision_tree(training_feature_array, training_label_array, testing_feature_array,
                                testing_label_array)
        class_report[Constant.DECISION_TREE] = Constant.DECISION_TREE
        class_report_list.append([class_report])

        class_report = \
            cls.__svm(training_feature_array, training_label_array, testing_feature_array, testing_label_array)
        class_report[Constant.SVM] = Constant.SVM
        class_report_list.append([class_report])

        class_report = \
            cls.__random_forest(training_feature_array, training_label_array, testing_feature_array,
                                testing_label_array)
        class_report[Constant.RANDOM_FOREST] = Constant.RANDOM_FOREST
        class_report_list.append([class_report])

        class_report = \
            cls.__ada_boost(training_feature_array, training_label_array, testing_feature_array, testing_label_array)
        class_report[Constant.ADA_BOOST] = Constant.ADA_BOOST
        class_report_list.append([class_report])

        class_report = \
            cls.__gradient_boost(training_feature_array, training_label_array, testing_feature_array,
                                 testing_label_array)
        class_report[Constant.GRADIENT_BOOST] = Constant.GRADIENT_BOOST
        class_report_list.append([class_report])

        class_report = \
            cls.__db_scan(testing_feature_array, testing_label_array)
        class_report[Constant.DB_SCAN] = Constant.DB_SCAN
        class_report_list.append([class_report])

        class_report = \
            cls.__gaussian_mixture(testing_feature_array, testing_label_array)
        class_report[Constant.GAUSSIAN_MIXTURE] = Constant.GAUSSIAN_MIXTURE
        class_report_list.append([class_report])

        class_report = \
            cls.__agglomerative_clustering(testing_feature_array, testing_label_array)
        class_report[Constant.AGGLOMERATIVE_CLUSTERING] = Constant.AGGLOMERATIVE_CLUSTERING
        class_report_list.append([class_report])

        cls.__calculate_scores(class_report_list)

    @classmethod
    def __calculate_total_candidate_score(cls):
        total_score_list = []

        ada_boost_score = sum(cls.__ada_boost_candidate_score_list)
        score_list = [ada_boost_score, Constant.ADA_BOOST]
        total_score_list.append(score_list)

        agglomerative_clustering_score = sum(cls.__agglomerative_clustering_candidate_score_list)
        score_list = [agglomerative_clustering_score, Constant.AGGLOMERATIVE_CLUSTERING]
        total_score_list.append(score_list)

        db_scan_score = sum(cls.__db_scan_candidate_score_list)
        score_list = [db_scan_score, Constant.DB_SCAN]
        total_score_list.append(score_list)

        decision_tree_score = sum(cls.__decision_tree_candidate_score_list)
        score_list = [decision_tree_score, Constant.DECISION_TREE]
        total_score_list.append(score_list)

        dnn_score = sum(cls.__dnn_candidate_score_list)
        score_list = [dnn_score, Constant.DNN]
        total_score_list.append(score_list)

        gaussian_mixture_score = sum(cls.__gaussian_mixture_candidate_score_list)
        score_list = [gaussian_mixture_score, Constant.GAUSSIAN_MIXTURE]
        total_score_list.append(score_list)

        gaussian_nb_score = sum(cls.__gaussian_nb_candidate_score_list)
        score_list = [gaussian_nb_score, Constant.GAUSSIAN_NB]
        total_score_list.append(score_list)

        gradient_boost_score = sum(cls.__gradient_boost_candidate_score_list)
        score_list = [gradient_boost_score, Constant.GRADIENT_BOOST]
        total_score_list.append(score_list)

        k_means_score = sum(cls.__k_means_candidate_score_list)
        score_list = [k_means_score, Constant.K_MEANS]
        total_score_list.append(score_list)

        knn_score = sum(cls.__knn_candidate_score_list)
        score_list = [knn_score, Constant.KNN]
        total_score_list.append(score_list)

        linear_regression_score = sum(cls.__linear_regression_candidate_score_list)
        score_list = [linear_regression_score, Constant.LINEAR_REGRESSION]
        total_score_list.append(score_list)

        linear_regression_lasso_score = sum(cls.__linear_regression_lasso_candidate_score_list)
        score_list = [linear_regression_lasso_score, Constant.LINEAR_REGRESSION_LASSO]
        total_score_list.append(score_list)

        linear_regression_ridge_score = sum(cls.__linear_regression_ridge_candidate_score_list)
        score_list = [linear_regression_ridge_score, Constant.LINEAR_REGRESSION_RIDGE]
        total_score_list.append(score_list)

        linear_regression_elastic_score = sum(cls.__linear_regression_elastic_candidate_score_list)
        score_list = [linear_regression_elastic_score, Constant.LINEAR_REGRESSION_ELASTIC]
        total_score_list.append(score_list)

        total_score_list = sorted(total_score_list, key=lambda a: a[0], reverse=True)
        return total_score_list[0][0], total_score_list[0][1]

    @classmethod
    def __save_scores(cls, score_list, save_path):
        if len(score_list) > 0:
            with open(save_path, 'w') as f:
                w = csv.writer(f)
                w.writerow(score_list[0].keys())
                for score in score_list:
                    w.writerow(score.values())

    @classmethod
    def ensemble_run(cls, training_feature_array, training_label_array, testing_feature_array, testing_label_array):
        score_list = []
        for count in range(0, 5):
            cls.__get_best_ml(training_feature_array, training_label_array, testing_feature_array,
                              testing_label_array)
            best_score, best_ml = cls.__calculate_total_candidate_score()
            score_list.append([best_score, best_ml])

        print('-----------------------------------')
        for index, score in enumerate(score_list):
            print(index, score)
        print('-----------------------------------')

        cls.__init_all_variables()
        cls.__save_scores(cls.__ada_boost_score_list, Constant.OUTPUT_DIR + '/' + Constant.ADA_BOOST + '_score.csv')
        cls.__save_scores(cls.__agglomerative_clustering_score_list, Constant.OUTPUT_DIR + '/' +
                          Constant.AGGLOMERATIVE_CLUSTERING + '_score.csv')
        cls.__save_scores(cls.__db_scan_score_list, Constant.OUTPUT_DIR + '/' + Constant.DB_SCAN + '_score.csv')
        cls.__save_scores(cls.__decision_tree_score_list, Constant.OUTPUT_DIR + '/'
                          + Constant.DECISION_TREE + '_score.csv')
        cls.__save_scores(cls.__dnn_score_list, Constant.OUTPUT_DIR + '/' + Constant.DNN + '_score.csv')
        cls.__save_scores(cls.__gaussian_mixture_score_list, Constant.OUTPUT_DIR + '/' +
                          Constant.GAUSSIAN_MIXTURE + '_score.csv')
        cls.__save_scores(cls.__gaussian_nb_score_list, Constant.OUTPUT_DIR + '/' + Constant.GAUSSIAN_NB + '_score.csv')
        cls.__save_scores(cls.__gradient_boost_score_list, Constant.OUTPUT_DIR + '/' +
                          Constant.GRADIENT_BOOST + '_score.csv')
        cls.__save_scores(cls.__k_means_score_list, Constant.OUTPUT_DIR + '/' + Constant.K_MEANS + '_score.csv')
        cls.__save_scores(cls.__knn_score_list, Constant.OUTPUT_DIR + '/' + Constant.KNN + '_score.csv')
        cls.__save_scores(cls.__linear_regression_score_list, Constant.OUTPUT_DIR + '/' +
                          Constant.LINEAR_REGRESSION + '_score.csv')
        cls.__save_scores(cls.__linear_regression_ridge_score_list, Constant.OUTPUT_DIR + '/' +
                          Constant.LINEAR_REGRESSION_RIDGE + '_score.csv')
        cls.__save_scores(cls.__linear_regression_lasso_score_list, Constant.OUTPUT_DIR + '/' +
                          Constant.LINEAR_REGRESSION_LASSO + '_score.csv')
        cls.__save_scores(cls.__linear_regression_elastic_score_list, Constant.OUTPUT_DIR + '/' +
                          Constant.LINEAR_REGRESSION_ELASTIC + '_score.csv')
        cls.__save_scores(cls.__random_forest_score_list, Constant.OUTPUT_DIR + '/' +
                          Constant.RANDOM_FOREST + '_score.csv')
        cls.__save_scores(cls.__svm_score_list, Constant.OUTPUT_DIR + '/' + Constant.SVM + '_score.csv')