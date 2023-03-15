import math
import random
import numpy as np
import Constant


class Combined_Feature_Engineering:
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

    @classmethod
    def make_same_feature_length(cls, first_feature_list, second_feature_list):
        first_len = len(first_feature_list)
        second_len = len(second_feature_list)
        temp_first_feature_list = None
        temp_second_feature_list = None
        same_size_flag = True
        largest_size = 0
        smallest_size = 0
        largest_list = None
        smallest_list = None
        first_flag = False

        if first_len > second_len:
            smallest_size = second_len
            largest_size = first_len
            largest_list = first_feature_list
            smallest_list = second_feature_list
            first_flag = True
        elif first_len < second_len:
            smallest_size = first_len
            largest_size = second_len
            largest_list = second_feature_list
            smallest_list = first_feature_list
        else:
            temp_first_feature_list = first_feature_list
            temp_second_feature_list = second_feature_list
            same_size_flag = False

        if same_size_flag:
            result = largest_size / smallest_size
            quotient = math.floor(result)
            remainder = result - quotient
            total_remainder = 0
            index = 0
            chosen_data_list = []

            while index < largest_size:
                if index % quotient == 0:
                    if total_remainder > quotient:
                        chosen_data_list.append(largest_list[index])
                        index += quotient
                        total_remainder = total_remainder - quotient
                    else:
                        total_remainder += remainder
                        chosen_data_list.append(largest_list[index])

                index += 1

            shrunk_index_size = len(chosen_data_list)
            index_list = list(index for index in range(0, shrunk_index_size))
            random_index_list = random.sample(index_list, smallest_size)
            random_index_list = sorted(random_index_list)

            final_shrunk_list = []
            for index in random_index_list:
                final_shrunk_list.append(chosen_data_list[index])

            if first_flag:
                temp_first_feature_list = final_shrunk_list
                temp_second_feature_list = smallest_list
            else:
                temp_first_feature_list = smallest_list
                temp_second_feature_list = final_shrunk_list

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
            np.array(list(int(features[last_index]) for features in training_feature_and_label_array))

        testing_feature_array = \
            np.array(list(list(map(float, features[:-1])) for features in testing_feature_and_label_array))

        testing_label_array = \
            np.array(list(int(features[last_index]) for features in testing_feature_and_label_array))

        return training_feature_array, training_label_array, testing_feature_array, testing_label_array

