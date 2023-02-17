import Constant


class STAT_Feature_Engineering_Single:
    def __init__(self, dataset):
        self.__dataset = dataset

    def __get_min_feature_len(self, feature_type):
        len_list = []
        feature_list = []
        for cs_id_feature_list in self.__dataset.get_normal_cs_stat_file_list():
            # cs_stat_cycle_list, cs_stat_instruction_list, cs_stat_branch_list
            for index, cs_id_feature in enumerate(cs_id_feature_list):
                len_list.append(len(cs_id_feature))
                if Constant.LIST_SEQUENCE[index] == feature_type:
                    feature_list.append(cs_id_feature)

        len_list = sorted(len_list)
        min_len = len_list[0]

        return min_len, feature_list

    def parsing_dataset(self, feature_type):
        min_len, feature_list = self.__get_min_feature_len(feature_type)
        temp_feature_list = []
        for single_feature in feature_list:
            temp_feature_list.append(single_feature[:min_len])

        for reduced_feature_list in temp_feature_list:
            print(len(reduced_feature_list))



