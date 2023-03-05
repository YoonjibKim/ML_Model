from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import Consensus


class Linear_Regressions:
    @classmethod
    def linear_regressions_run(cls, X_tn, y_tn, X_te, y_te):
        # 데이터 표준화
        std_scale = StandardScaler()
        std_scale.fit(X_tn)
        X_tn_std = std_scale.transform(X_tn)
        X_te_std = std_scale.transform(X_te)

        # 선형 회귀분석 학습
        clf_lr = LinearRegression()
        clf_lr.fit(X_tn_std, y_tn)

        # 선형 회귀분석 모형 추정 계수 확인
        print(clf_lr.coef_)
        print(clf_lr.intercept_)

        # 릿지 회귀분석(L2 제약식 적용)
        clf_ridge = Ridge(alpha=1)
        clf_ridge.fit(X_tn_std, y_tn)

        # 릿지 회귀분석 모형 추정 계수 확인
        print(clf_ridge.coef_)
        print(clf_ridge.intercept_)

        # 라쏘 회귀분석(L1 제약식 적용)
        clf_lasso = Lasso(alpha=0.01)
        clf_lasso.fit(X_tn_std, y_tn)

        # 라쏘 회귀분석 모형 추정 계수 확인
        print(clf_lasso.coef_)
        print(clf_lasso.intercept_)

        # 엘라스틱넷
        clf_elastic = ElasticNet(alpha=0.01, l1_ratio=0.01)
        clf_elastic.fit(X_tn_std, y_tn)

        # 엘라스틱넷 모형 추정 계수 확인
        print(clf_elastic.coef_)
        print(clf_elastic.intercept_)

        # 예측
        pred_lr = clf_lr.predict(X_te_std)
        pred_ridge = clf_ridge.predict(X_te_std)
        pred_lasso = clf_lasso.predict(X_te_std)
        pred_elastic = clf_elastic.predict(X_te_std)

        # 모형 평가-R제곱값
        print(r2_score(y_te, pred_lr))
        print(r2_score(y_te, pred_ridge))
        print(r2_score(y_te, pred_lasso))
        print(r2_score(y_te, pred_elastic))

        # 모형 평가-MSE
        print(mean_squared_error(y_te, pred_lr))
        print(mean_squared_error(y_te, pred_ridge))
        print(mean_squared_error(y_te, pred_lasso))
        print(mean_squared_error(y_te, pred_elastic))
