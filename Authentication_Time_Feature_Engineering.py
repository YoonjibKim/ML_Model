import csv

import Constant


class Authentication_Time_Feature_Engineering:
    def __init__(self, scenario, random_cs_mode, gaussian_mode):
        root_path = Constant.BEFORE_PARSING_DATASET_PATH
        default_path = root_path + '/' + scenario + '/' + random_cs_mode + '/' + gaussian_mode
        attack_path = default_path + '/' + Constant.ATTACK
        normal_path = default_path + '/' + Constant.NORMAL

        attack_authentication_results_path = attack_path + '/' + Constant.AUTHENTICATION_RESULTS_FILE_NAME
        normal_authentication_results_path = normal_path + '/' + Constant.AUTHENTICATION_RESULTS_FILE_NAME

        attack_authentication_results_file = self.__read_csv(attack_authentication_results_path)
        normal_authentication_results_file = self.__read_csv(normal_authentication_results_path)


    @classmethod
    def __read_csv(cls, path):
        read_line = []
        with open(path, 'r') as f:
            rdr = csv.reader(f)
            for line in rdr:
                read_line.append(line)

        return read_line
