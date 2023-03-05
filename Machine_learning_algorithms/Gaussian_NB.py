from sklearn.metrics import recall_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


class Gaussian_NB:
    @classmethod
    def gaussian_nb_run(cls, training_feature_array, training_label_array, testing_feature_array,
                        testing_label_array):
        # 데이터 표준화
        std_scale = StandardScaler()
        std_scale.fit(training_feature_array)
        X_tn_std = std_scale.transform(training_feature_array)
        X_te_std = std_scale.transform(testing_feature_array)

        # 나이브 베이즈 학습
        clf_gnb = GaussianNB()
        clf_gnb.fit(X_tn_std, training_label_array.ravel())

        # 예측
        pred_gnb = clf_gnb.predict(X_te_std)
        print(pred_gnb)

        # 리콜
        recall = recall_score(testing_label_array, pred_gnb, average='macro')
        print(recall)

        # confusion matrix 확인
        conf_matrix = confusion_matrix(testing_label_array, pred_gnb)
        print(conf_matrix)

        # 분류 레포트 확인
        class_report = classification_report(testing_label_array, pred_gnb)
        print(class_report)
