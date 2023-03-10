import math


class APRF:
    @classmethod
    def __arf(cls, omega, beta):
        square_coefficient = math.sqrt(2)
        apf_val = omega * square_coefficient * (math.exp(beta) - 1)

        return apf_val

    @classmethod
    def __apf(cls, omega, beta, alpha):
        omega_coefficient = math.sqrt(1 + (omega ** 2)) / omega
        exp_val = math.exp(omega * beta) - math.exp(omega * (beta - alpha))
        arf_val = omega_coefficient * exp_val

        return arf_val

    @classmethod
    def aprf_run(cls, positive_score, negative_score):
        positive_apf = cls.__arf(1.0, positive_score)
        negative_arf = cls.__apf(1.5, positive_score, negative_score)

        total_score = positive_apf - negative_arf

        return total_score
