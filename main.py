#!/usr/bin/python3

import sklearn
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.linear_model
import sklearn.ensemble
import sklearn.neighbors
import sklearn.mixture
import sklearn.metrics
import numpy as np
import argparse

import modules
import utils
import config

def main(fname: str):
    # Read data
    features_header, identifier, features_str, label_str = utils.read_csv(fname, config.FEATURE_COLS, config.LABEL_COL)

    # Feature Preprocessing
    features = np.array(list(map(lambda x: utils.process_features(x, feature_map_fcn = {
        41: lambda x: 1 if x == '1' else 0,
        42: lambda x: np.char.count(x, '1'),
        51: lambda x: utils.process_prov_feature(x, config.PROV_FEATURE_TYPE_1),
        52: lambda x: utils.process_prov_feature(x, config.PROV_FEATURE_TYPE_1),
        53: lambda x: utils.process_prov_feature(x, config.PROV_FEATURE_TYPE_1),
        54: lambda x: utils.process_prov_feature(x, config.PROV_FEATURE_TYPE_2)
    }), features_str)))
    label = np.array(list(map(lambda y: utils.process_label(y, config.LABEL_DICT, config.UNLABELED), label_str)))

    # Dataset Split
    pt_idx, train_l_idx, train_u_idx, test_idx = utils.dataset_split(identifier, label, config.UNLABELED)
    identifier_pt = identifier[pt_idx]
    features_pt = features[pt_idx]
    label_pt = label[pt_idx]

    identifier_train_l = identifier[train_l_idx]
    features_train_l = features[train_l_idx]
    label_train_l = label[train_l_idx]
    identifier_train_u = identifier[train_u_idx]
    features_train_u = features[train_u_idx]

    identifier_test = identifier[test_idx]
    features_test = features[test_idx]
    label_test = label[test_idx]

    print(f'PreTraining Dataset Size: {np.shape(features_pt), np.shape(label_pt)}')
    print(f'Training Dataset Size: {np.shape(features_train_l), np.shape(label_train_l)}, {np.shape(features_train_u)}')
    print(f'Test Dataset Size: {np.shape(features_test), np.shape(label_test)}')

    # Feature Selection
    feature_selected = utils.select_feature(features_pt, label_pt)

    # Cross Validation

    ## Naive Bayes
    # clf = modules.SemiSupervisedLearningModel(sklearn.naive_bayes.GaussianNB(), config.UNLABELED, use_ssl = False)

    ## Logistic Regression
    # clf = modules.SemiSupervisedLearningModel(sklearn.linear_model.LogisticRegression(C = 1e3, penalty = 'l2', solver = 'liblinear'), config.UNLABELED, use_ssl = False)

    ## Random Forest
    # clf = modules.SemiSupervisedLearningModel(sklearn.ensemble.RandomForestClassifier(n_estimators = 300, max_depth = 35, n_jobs = -1), config.UNLABELED, use_ssl = False)

    ## AdaBoost
    # clf = modules.SemiSupervisedLearningModel(sklearn.ensemble.AdaBoostClassifier(n_estimators = 400), config.UNLABELED, use_ssl = False)

    ## k-NN
    # clf = modules.SemiSupervisedLearningModel(sklearn.neighbors.KNeighborsClassifier(n_neighbors = 6, n_jobs = -1), config.UNLABELED, use_ssl = False)

    ## Propagating k-NN
    # clf = modules.SemiSupervisedLearningModel(SelfTrainingClassifier(
    #     sklearn.neighbors.KNeighborsClassifier(n_neighbors = 10, n_jobs = -1), n_soft_labels = 100), config.UNLABELED)

    ## Cluster-Then-Label
    n_clusters = 3
    clus, clf_array = sklearn.mixture.GaussianMixture(n_components = n_clusters, max_iter = 20), []
    for i in range(n_clusters):
        clf_array.append(sklearn.ensemble.RandomForestClassifier(n_estimators = 300, max_depth = 35, n_jobs = -1))
    clf = modules.SemiSupervisedLearningModel(modules.ClusterThenLabelModel(config.UNLABELED, clus, n_clusters, clf_array, True), config.UNLABELED)

    acc_cv = utils.cross_validation(5, clf, features_train_l[:, feature_selected], label_train_l, identifier_train_l,
        features_train_u[:, feature_selected])
    print(f'Cross Validation Accuracy: {acc_cv}')

    # Test
    clf.fit(features_train_l[:, feature_selected], label_train_l, features_train_u[:, feature_selected])
    cm = sklearn.metrics.confusion_matrix(label_test, clf.predict(features_test[:, feature_selected]), labels = [config.POS_LABEL, config.NEG_LABEL])
    precision, accuracy = cm[0, 0] / np.sum(cm[:, 0]), np.trace(cm) / np.sum(cm)
    print(f'Test Precision: {precision}')
    print(f'Test Accuracy: {accuracy}')
    label_proba = clf.predict_proba(features_test[:, feature_selected])
    label_score = np.squeeze(label_proba[:, clf.classes_ == config.POS_LABEL])
    auc = sklearn.metrics.roc_auc_score(label_test, label_score)
    print(f'AUC: {auc}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help = 'data file')
    args = parser.parse_args()

    fname = args.data
    main(fname)
