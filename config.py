import numpy as np

FEATURE_COLS, LABEL_COL = np.concatenate([np.arange(3, 69), np.arange(70, 99)]), np.array([69])

FEATURE_SIZE = len(FEATURE_COLS)

POS_LABEL, NEG_LABEL, UNLABELED = 1, 0, -1

LABEL_DICT = {
    'PC': POS_LABEL,
    'AFP': NEG_LABEL,
    'NTP': NEG_LABEL
}

PROV_FEATURE_TYPE_1 = ['KIC', 'SPE', 'PHO', 'TRA', 'AST', 'Solar']
PROV_FEATURE_TYPE_2 = ['DSEP', 'MULT']
