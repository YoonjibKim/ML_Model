from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class KNN:
    @classmethod
    def knn_run(cls, training_data_array, training_label_array, testing_data_array, testing_label_array):
        std_scale = StandardScaler()
        std_scale.fit(training_data_array)

        X_tn_std = std_scale.transform(training_data_array)
        X_te_std = std_scale.transform(testing_data_array)

        clf_knn = KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='kd_tree', p=2)
        clf_knn.fit(X_tn_std, training_label_array)
        knn_pred = clf_knn.predict(X_te_std)

        class_report = classification_report(testing_label_array, knn_pred)
        print(class_report)