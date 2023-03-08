class APRF:
    @classmethod
    def aprf_run(cls, normal_count, attack_count, f1_score):
        total_count = normal_count + attack_count
        positive_ratio = normal_count / total_count
        negative_ratio = attack_count / total_count
        positive_score = f1_score * positive_ratio
        negative_score = (1.0 - f1_score) * negative_ratio

        score = cls.__calculate_penalty_and_reward(positive_score, negative_score)
        return score

    @classmethod
    def __calculate_penalty_and_reward(cls, positive_score, negative_score):
        return 100
