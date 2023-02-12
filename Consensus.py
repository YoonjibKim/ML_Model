from Machine_learning_algorithms.K_Nearest_Neighbor import KNN


class Consensus(KNN):
    def __init__(self, training_normal_data_dict, training_attack_data_dict, testing_normal_data, testing_attack_data):
        self.__training_normal_data_dict = training_normal_data_dict
        self.__training_attack_data_dict = training_attack_data_dict
        self.__testing_normal_data = testing_normal_data
        self.__testing_normal_data = testing_attack_data
        
    def knn(self):
        KNN.run(self, self.__training_normal_data_dict, self.__training_attack_data_dict, self.__testing_normal_data,
                self.__testing_normal_data)
