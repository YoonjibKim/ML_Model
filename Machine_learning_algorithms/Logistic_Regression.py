from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


class Logistic_Regression:
    @classmethod
    def logistic_regression_run(cls, training_feature_array, training_label_array, testing_feature_array,
                                testing_label_array):
        std_scale = StandardScaler()
        std_scale.fit(training_feature_array)
        X_tn_std = std_scale.transform(training_feature_array)
        X_te_std = std_scale.transform(testing_feature_array)

        clf_logi_l2 = LogisticRegression(penalty='l2')
        clf_logi_l2.fit(X_tn_std, training_label_array.ravel())

        # 로지스틱 회귀분석 모형(L2 제약식 적용) 추정 계수
        print(clf_logi_l2.coef_)
        print(clf_logi_l2.intercept_)

        # 예측
        pred_logistic = clf_logi_l2.predict(X_te_std)
        print(pred_logistic)

        # 확률값으로 예측
        pred_proba = clf_logi_l2.predict_proba(X_te_std)
        print(pred_proba)

        # 정밀도
        precision = precision_score(testing_label_array, pred_logistic)
        print(precision)

        # confusion matrix 확인
        conf_matrix = confusion_matrix(testing_label_array, pred_logistic)
        print(conf_matrix)

        # 분류 레포트 확인
        class_report = classification_report(testing_label_array, pred_logistic)
        print(class_report)
