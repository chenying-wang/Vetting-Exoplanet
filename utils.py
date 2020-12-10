import sklearn
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.feature_selection
import numpy as np

def read_csv(fname: str, features_cols: np.ndarray, label_cols: np.ndarray, delimiter: str = ',') -> tuple:
    print(f'Input Data File: \'{fname}\'')
    features_header = np.genfromtxt(fname, dtype = np.str, delimiter = delimiter, max_rows = 1, usecols = features_cols)
    identifier = np.genfromtxt(fname, dtype = np.int, delimiter = delimiter, usecols = [1, 2])[1:]
    features_str = np.genfromtxt(fname, dtype = np.str, delimiter = delimiter, usecols = features_cols)[1:]
    label_str = np.genfromtxt(fname, dtype = np.str, delimiter = delimiter, usecols = label_cols)[1:]
    print(f'Dataset Size: {np.shape(features_str), np.shape(label_str)}')
    return features_header, identifier, features_str, label_str

def process_prov_feature(prov: str, prov_array: list) -> int:
    for i in range(len(prov_array)):
        if prov.startswith(prov_array[i]):
            return i + 1
    return 0

def process_features(features_str: np.ndarray, feature_map_fcn: dict = {}) -> np.ndarray:
    x = np.empty_like(features_str, dtype = np.float)
    for i in range(len(features_str)):
        x[i] = feature_map_fcn[i](features_str[i]) if i in feature_map_fcn else float(features_str[i])
    return x

def process_label(label_str: str, label_dict: dict, unlabeled: int) -> int:
    if label_str in label_dict:
        return label_dict[label_str]
    return unlabeled


def dataset_split(identifier: np.ndarray, label: np.ndarray, unlabeled: int) -> tuple:
    # Train/Test Split
    rng = np.random.default_rng(seed = 42)
    group_l = np.setdiff1d(np.unique(identifier[:, 0]), np.unique(identifier[label == unlabeled, 0]))
    _, group_test = sklearn.model_selection.train_test_split(group_l, test_size = 0.3, random_state = 2)
    group_test = set(group_test)

    train_idx = [g not in group_test for g in identifier[:, 0]]
    train_l_idx = np.nonzero(np.logical_and(train_idx, label != unlabeled))[0]
    train_u_idx = np.nonzero(np.logical_and(train_idx, label == unlabeled))[0]
    test_idx = np.nonzero(np.logical_not(train_idx))[0]

    train_l_size = len(train_l_idx)
    if train_l_size < len(train_l_idx):
        rand_idx = rng.permutation(len(train_l_idx))
        train_u_idx = np.append(train_u_idx, train_l_idx[rand_idx[:-train_l_size]])
        train_l_idx = train_l_idx[rand_idx[-train_l_size:]]
        del rand_idx

    rng.shuffle(train_l_idx)
    rng.shuffle(train_u_idx)
    rng.shuffle(test_idx)

    rand_idx = rng.permutation(len(train_l_idx))
    train_pt_idx = rand_idx[:int(len(train_l_idx) * 0.1)]
    train_tr_idx = rand_idx[int(len(train_l_idx) * 0.1):]

    pt_idx = train_l_idx[train_pt_idx]
    train_l_idx = train_l_idx[train_tr_idx]
    return pt_idx, train_l_idx, train_u_idx, test_idx

def select_feature(features: np.ndarray, label: np.ndarray) -> np.ndarray:
    scaler = sklearn.preprocessing.StandardScaler()
    features_std = scaler.fit_transform(features)

    clf = sklearn.linear_model.LogisticRegression(C = 10, penalty = 'l1', solver = 'saga')
    selector = sklearn.feature_selection.SelectFromModel(clf, threshold = 1e-9, max_features = 60).fit(features_std, label)
    print(f'Total {np.sum(selector.get_support())} Features Selected')
    return selector.get_support()

def cross_validation(n_splits: int, clf: object, features_l: np.ndarray, label_l: np.ndarray, identifier_l: np.ndarray, features_u: np.ndarray = None) -> float:
    k_fold = sklearn.model_selection.GroupKFold(n_splits = n_splits)
    acc_val = []
    for cv_train_idx, cv_val_idx in k_fold.split(X = features_l, y = label_l, groups = identifier_l[:, 0]):
        clf.fit(features_l[cv_train_idx], label_l[cv_train_idx], features_u)
        acc_val.append(clf.score(features_l[cv_val_idx], label_l[cv_val_idx]))
    return np.mean(acc_val)
