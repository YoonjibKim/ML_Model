from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


class Gradient_Boost:
    @classmethod
    def gradient_boost_run(cls, X_tn, y_tn, X_te, y_te):
        std_scale = StandardScaler()
        std_scale.fit(X_tn)
        X_tn_std = std_scale.transform(X_tn)
        X_te_std = std_scale.transform(X_te)

        clf_gbt = GradientBoostingClassifier(max_depth=2, learning_rate=0.01, random_state=0)
        clf_gbt.fit(X_tn_std, y_tn.ravel())

        pred_gboost = clf_gbt.predict(X_te_std)

        class_report = classification_report(y_te, pred_gboost, output_dict=True, zero_division=0)

        return class_report
