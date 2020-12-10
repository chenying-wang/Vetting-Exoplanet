import sklearn
import sklearn.preprocessing
import sklearn.metrics
import numpy as np

class BaseModel:

    def __init__(self, clf: object, unlabeled: int, proba: bool = True):
        self.clf = clf
        self.unlabeled = unlabeled
        self.proba = proba

    def fit(self, features, label):
        self.clf.fit(features, label)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, features):
        return self.clf.predict(features)

    def predict_proba(self, features):
        if self.proba:
            return self.clf.predict_proba(features)
        label = self.clf.predict(features)
        proba = np.zeros((len(features), len(self.classes_)), dtype = np.float)
        for i in range(len(self.classes_)):
            proba[:, i] = np.where(label == self.classes_[i], 1, 0)
        return proba

    def score(self, features, label):
        return self.clf.score(features, label)


class SelfTrainingClassifier(BaseModel):

    def __init__(self, clf: object, unlabeled: int, n_soft_labels: int = 100):
        super().__init__(clf, unlabeled)
        self.n_soft_labels = n_soft_labels

    def fit(self, features: np.ndarray, label: np.ndarray):
        idx_mask = np.arange(len(label))
        while np.count_nonzero(label != self.unlabeled) < len(label):
            self.clf.fit(features[label != self.unlabeled], label[label != self.unlabeled])
            prob = self.clf.predict_proba(features[label == self.unlabeled])

            new_label = label.copy()
            for i in range(self.n_soft_labels):
                max_prob_idx = np.unravel_index(np.argmax(prob), np.shape(prob))
                idx = idx_mask[label == self.unlabeled][max_prob_idx[0]]
                new_label[idx] = self.clf.classes_[max_prob_idx[1]]
                if np.count_nonzero(new_label == self.unlabeled) == 0:
                    break
                prob[max_prob_idx[0]] = 0
            del label
            label = new_label

        self.classes_ = self.clf.classes_
        return self

class ClusterThenLabelModel(BaseModel):

    def __init__(self, unlabeled: int, clus: object, n_clusters: int, clf_array: list, mixture: bool = False):
        super().__init__(None, unlabeled)
        self.n_clusters = n_clusters
        self.clus = clus
        self.clf_array = clf_array
        self.mixture = mixture

    def fit(self, features: np.ndarray, label: np.ndarray):
        label_clus = self.clus.fit_predict(features)

        self.classes_ = np.unique(label[label != self.unlabeled])
        for i in range(self.n_clusters):
            idx = np.logical_and(label_clus == i, label != self.unlabeled)
            if len(np.unique(label[idx])) < 2:
                idx = label != self.unlabeled
            self.clf_array[i].fit(features[idx], label[idx])

    def predict(self, features):
        proba = self.predict_proba(features)
        return np.choose(np.argmax(proba, axis = 1), self.classes_)

    def predict_proba(self, features):
        label_clf_proba_array = []
        for i in range(self.n_clusters):
            label_clf_proba_array.append(self.clf_array[i].predict_proba(features))

        proba = np.zeros((len(features), len(self.classes_)), dtype = np.float)
        if self.mixture:
            label_clus_proba = self.clus.predict_proba(features)
            for i in range(len(self.classes_)):
                for j in range(self.n_clusters):
                    proba[:, i] += label_clus_proba[:, j] * \
                        np.squeeze(label_clf_proba_array[j][:, self.clf_array[i].classes_ == self.classes_[i]])
            return proba

        label_clus = self.clus.predict(features)
        for i in range(len(self.classes_)):
            for j in range(self.n_clusters):
                proba[:, i] += np.where(label_clus == j,
                    np.squeeze(label_clf_proba_array[j][:, self.clf_array[i].classes_ == self.classes_[i]]), 0)
        return proba

    def score(self, features, label):
        return sklearn.metrics.accuracy_score(label, self.predict(features))

class SemiSupervisedLearningModel(BaseModel):

    def __init__(self, clf: object, unlabeled: int, use_ssl: bool = True, proba: bool = True):
        super().__init__(clf, unlabeled, proba)
        self.use_ssl = use_ssl
        self.scaler = sklearn.preprocessing.StandardScaler()

    def fit(self, features_l: np.ndarray, label_l: np.ndarray, features_u: np.ndarray = None):
        if self.use_ssl and features_u is not None:
            self.scaler.fit(np.vstack((features_l, features_u)))
            features_u_std = self.scaler.transform(features_u)
        else:
            self.scaler.fit(features_l)
        features_l_std = self.scaler.transform(features_l)

        if self.use_ssl and features_u is not None:
            # Semi-Supervised Learning
            self.clf.fit(np.vstack((features_l_std, features_u_std)),
                np.append(label_l, np.full(len(features_u_std), self.unlabeled)))
        else:
            # Supervised Learning
            self.clf.fit(features_l_std, label_l)
        self.classes_ = self.clf.classes_
        return self

    def predict(self, features):
        return super().predict(self.scaler.transform(features))

    def predict_proba(self, features):
        return super().predict_proba(self.scaler.transform(features))

    def score(self, features, label):
        return super().score(self.scaler.transform(features), label)
