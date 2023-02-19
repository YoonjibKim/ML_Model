import numpy as np

import Constant


class STAT_Feature_Engineering_Single:
    __row_size = 0
    __col_size = 0

    @classmethod
    def __get_min_feature_len(cls, feature_type, cs_stat_file_list):
        len_list = []
        feature_list = []
        for cs_id_feature_list in cs_stat_file_list:
            # cs_stat_cycle_list, cs_stat_instruction_list, cs_stat_branch_list
            for index, sub_cs_id_feature_list in enumerate(cs_id_feature_list):
                len_list.append(len(sub_cs_id_feature_list))
                if Constant.LIST_SEQUENCE[index] == feature_type:
                    int_cs_id_feature_list = []
                    for feature in sub_cs_id_feature_list:
                        int_cs_id_feature_list.append(int(feature.replace(',', '')))
                    feature_list.append(int_cs_id_feature_list)

        len_list = sorted(len_list)
        min_len = len_list[0]

        return min_len, feature_list

    def parsing_dataset(self, feature_type, cs_stat_file_list):
        min_len, feature_list = self.__get_min_feature_len(feature_type, cs_stat_file_list)
        temp_feature_list = []
        for single_feature in feature_list:
            temp_feature_list.append(single_feature[:min_len])

        self.__col_size = min_len
        self.__row_size = len(temp_feature_list)
        combined_array = np.empty(shape=(self.__row_size,), dtype=int)
        for index in range(0, min_len):
            record_list = []
            for sub_feature_list in temp_feature_list:
                data = sub_feature_list[index]
                record_list.append(data)
            record_array = np.array(record_list)
            combined_array = np.vstack((combined_array, record_array))

        combined_array = combined_array[1:]

        return combined_array

    def get_label(self, feature_type):
        label_array = None
        if feature_type == Constant.ATTACK:
            label_array = np.round(np.ones([self.__col_size, self.__row_size])).astype(int)
        elif feature_type == Constant.NORMAL:
            label_array = np.round(np.zeros([self.__col_size, self.__row_size])).astype(int)

        return label_array
