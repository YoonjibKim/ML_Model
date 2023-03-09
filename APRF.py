import math
import numpy as np
from scipy.special import erfi


class APRF:
    @classmethod
    def aprf_run(cls, count, normal_count, attack_count, f1_score):
        total_count = normal_count + attack_count
        positive_ratio = normal_count / total_count
        negative_ratio = attack_count / total_count
        positive_score = f1_score * positive_ratio
        negative_score = (1.0 - f1_score) * negative_ratio

        score = cls.__calculate_penalty_and_reward(count, positive_score, negative_score)
        return score

    @classmethod
    def __apf(cls, n, omega, beta):
        sigma_val = 0
        for i in range(0, n + 1):
            temp = 1 + 4 * (beta * beta)
            sqrt_val = math.sqrt(temp)
            sigma_val += (erfi(beta) * (sqrt_val - 2 * np.log(sqrt_val)))

        apf_val = ((omega * math.sqrt(math.pi)) / 2) * sigma_val

        return apf_val

    @classmethod
    def __arf(cls, n, omega, beta):
        return 0

    @classmethod
    def __calculate_penalty_and_reward(cls, count, positive_score, negative_score):
        positive_apf = cls.__apf(count, 1.0, positive_score)
        negative_arf = cls.__arf(count, 1.0, negative_score)
        total_score = positive_apf + negative_arf

        return total_score
