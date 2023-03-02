import numpy as np

import Constant


class Combined_Feature_Engineering:
    @classmethod
    def __get_stat_and_time_diff_combined_features(cls, first_mixed_list, second_mixed_list):

        first_list_size = len(first_mixed_list)
        second_list_size = len(second_mixed_list)
        large_list = None
        small_list = None
        diff_size = None
        flag_1 = False
        flag_2 = False

        if first_list_size > second_list_size:
            diff_size = first_list_size - second_list_size
            large_list = first_mixed_list
            small_list = second_mixed_list
            flag_1 = True
        elif first_list_size < second_list_size:
            diff_size = second_list_size - first_list_size
            large_list = second_mixed_list
            small_list = first_mixed_list
            flag_2 = True

        first_list = None
        second_list = None

        if flag_1 or flag_2:
            large_list_size = len(large_list)
            list_ratio = large_list_size / diff_size
            round_ratio = round(list_ratio)
            del large_list[::round_ratio]

            if flag_1:
                first_list = large_list
                second_list = small_list
            else:
                first_list = small_list
                second_list = large_list

        return first_list, second_list

    @classmethod
    def divide_attack_and_normal_features(cls, mixed_list):
        attack_list = []
        normal_list = []
        label_pos = len(mixed_list[0]) - 1
        for data in mixed_list:
            label = data[label_pos]
            if int(label) == Constant.NORMAL_LABEL:
                normal_list.append(data)
            else:
                attack_list.append(data)

        return attack_list, normal_list

    def make_same_feature_length(self, first_feature_list, second_feature_list):
        while True:
            small_list, large_list = self.__get_stat_and_time_diff_combined_features(first_feature_list,
                                                                                     second_feature_list)
            small_list_size = len(small_list)
            large_list_size = len(large_list)
            if small_list_size == large_list_size:
                temp_first_feature_list = small_list
                temp_second_feature_list = large_list
                break

        return temp_first_feature_list, temp_second_feature_list

    @classmethod
    def __combine_features_to_array(cls, first_feature_list, second_feature_list):
        stat_attack_feature_array = np.array(list(data[:-1] for data in first_feature_list))
        time_diff_attack_feature_array = np.array(list([data[0]] for data in second_feature_list))
        attack_label_array = np.array(list([data[1]] for data in second_feature_list))

        combined_attack_feature_array = np.append(stat_attack_feature_array, time_diff_attack_feature_array, axis=1)
        combined_attack_feature_array = np.append(combined_attack_feature_array, attack_label_array, axis=1)

        return combined_attack_feature_array

    def get_training_and_testing_feature_array(self, stat_mixed_attack_list, time_diff_mixed_attack_list,
                                               stat_mixed_normal_list, time_diff_mixed_normal_list):
        combined_attack_feature_and_label_array = \
            self.__combine_features_to_array(stat_mixed_attack_list, time_diff_mixed_attack_list)
        combined_normal_feature_and_label_array = \
            self.__combine_features_to_array(stat_mixed_normal_list, time_diff_mixed_normal_list)

        combined_feature_and_label_array = np.append(combined_attack_feature_and_label_array,
                                                     combined_normal_feature_and_label_array, axis=0)

        np.random.shuffle(combined_feature_and_label_array)

        total_size = len(combined_feature_and_label_array)
        training_size = total_size * Constant.TRAINING_FEATURE_RATIO
        training_size = round(training_size)
        testing_size = total_size - training_size

        training_feature_and_label_array = combined_feature_and_label_array[0:training_size]
        testing_feature_and_label_array = combined_feature_and_label_array[-testing_size:]

        last_index = len(testing_feature_and_label_array[0]) - 1
        training_feature_array = \
            np.array(list(list(map(float, features[:-1])) for features in training_feature_and_label_array))
        training_label_array = \
            np.array(list([int(features[last_index])] for features in training_feature_and_label_array))

        testing_feature_array = \
            np.array(list(list(map(float, features[:-1])) for features in testing_feature_and_label_array))

        testing_label_array = \
            np.array(list([int(features[last_index])] for features in testing_feature_and_label_array))

        return training_feature_array, training_label_array, testing_feature_array, testing_label_array

