import Constant
from CS_Implementation import CS_Implementation
from Consensus import Consensus
from GS_Implementation import GS_Implementation


def calculate_feature_size(param_feature_array):
    normal_count = 0
    attack_count = 0
    for i in param_feature_array:
        if i == Constant.NORMAL_LABEL:
            normal_count += 1
        else:
            attack_count += 1

    return normal_count, attack_count


def output_feature_information(param_training_label_array, param_testing_label_array):
    ret_normal_count, ret_attack_count = calculate_feature_size(param_training_label_array)
    print('training normal count: ', ret_normal_count)
    print('training attack count: ', ret_attack_count)
    print('training feature dimension: ', training_feature_array.shape[1])
    print('training total count: ', training_feature_array.shape[0])
    ret_normal_count, ret_attack_count = calculate_feature_size(param_testing_label_array)
    print('testing normal count: ', ret_normal_count)
    print('testing attack count: ', ret_attack_count)
    print('testing feature dimension:', testing_feature_array.shape[1])
    print('testing total count: ', testing_feature_array.shape[0])


def gs_implementation(param_attack_scenario):
    gs = GS_Implementation(param_attack_scenario)
    gs.top_feature_analysis()

    gs_training_feature_array, gs_training_label_array, gs_testing_feature_array, \
        gs_testing_label_array = gs.get_top_cycle_feature_and_label_array()

    gs_training_feature_array, gs_training_label_array, gs_testing_feature_array, \
        gs_testing_label_array = gs.get_top_instructions_feature_and_label_array()

    gs_training_feature_array, gs_training_label_array, gs_testing_feature_array, \
        gs_testing_label_array = gs.get_top_branch_feature_and_label_array()

    gs.stat_feature_analysis()

    return gs_training_feature_array, gs_training_label_array, gs_testing_feature_array, gs_testing_label_array


def cs_implementation(param_attack_scenario):
    cs = CS_Implementation(param_attack_scenario)
    cs.stat_feature_analysis()

    cs_training_feature_array, cs_training_label_array, cs_testing_feature_array, cs_testing_label_array = \
        cs.get_stat_cycle_feature_and_label_array()

    # cs_training_feature_array, cs_training_label_array, cs_testing_feature_array, cs_testing_label_array = \
    #     cs.get_time_diff_feature_and_label_array()

    # cs_training_feature_array, cs_training_label_array, cs_testing_feature_array, cs_testing_label_array = \
    #     cs.get_stat_time_diff_feature_and_label_array()

    # cs_training_feature_array, cs_training_label_array, cs_testing_feature_array, cs_testing_label_array = \
    #     cs.get_top_stat_time_diff_feature_and_label_array()
    #
    # cs_training_feature_array, cs_training_label_array, cs_testing_feature_array, cs_testing_label_array = \
    #     cs.get_shrunk_feature_array()

    # cs_training_feature_array, cs_training_label_array, cs_testing_feature_array, cs_testing_label_array = \
    #     cs.get_extended_feature_array()

    return cs_training_feature_array, cs_training_label_array, cs_testing_feature_array, cs_testing_label_array


if __name__ == '__main__':
    print('Simulation Start')

    attack_scenario = [Constant.CORRECT_EV_ID, Constant.RANDOM_CS_ON, Constant.GAUSSIAN_OFF]

    # training_feature_array, training_label_array, testing_feature_array, testing_label_array = \
    #     cs_implementation(attack_scenario)

    training_feature_array, training_label_array, testing_feature_array, testing_label_array = \
        gs_implementation(attack_scenario)

    Consensus.ensemble_run(training_feature_array, training_label_array, testing_feature_array, testing_label_array)
    output_feature_information(training_label_array, testing_label_array)

    print('Simulation End')
