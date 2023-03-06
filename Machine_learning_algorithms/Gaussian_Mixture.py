from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture


class Gaussian_Mixture:
    @classmethod
    def gaussian_mixture_run(cls, X, y):
        gmm = GaussianMixture(n_components=2, random_state=0)
        gmm.fit(X)
        label_gmm = gmm.predict(X)

        class_report = classification_report(y, label_gmm, output_dict=True, zero_division=0)

        return class_report
