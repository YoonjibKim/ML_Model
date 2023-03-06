from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report


class DB_Scan:
    @classmethod
    def db_scan_run(cls, X, y):
        dbs = DBSCAN(eps=250000000)
        dbs.fit(X)
        label_dbs = dbs.labels_

        class_report = classification_report(y, label_dbs, output_dict=True, zero_division=0)

        return class_report
