from sklearn.cluster import KMeans
from sklearn.metrics import classification_report


class K_Means:
    @classmethod
    def k_means_run(cls, testing_data_array, testing_label_array):
        kmc = KMeans(n_clusters=2,
                     init='random',
                     n_init='auto',
                     max_iter=100,
                     random_state=0)
        kmc.fit(testing_data_array)
        label_kmc = kmc.labels_

        class_report = classification_report(testing_label_array, label_kmc, output_dict=True, zero_division=0)
        weighted_avg = class_report['weighted avg']
        f1_score = weighted_avg['f1-score']
        print('f1-score: ', f1_score)