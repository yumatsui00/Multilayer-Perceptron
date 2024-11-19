import numpy as np
from icecream import ic

def standarization(data):
    #標準化するよ
    for feature in [col for col in data.columns if col != "Diagnosis"]:
        data[feature] = (data[feature] - data[feature].mean()) / data[feature].std()
    return data


def split_train_test(data, test_size=0.3, random_state=None):
    if (random_state):
        np.random.seed(random_state)
    p = np.random.permutation(len(data.index))
    testsize = int(len(data.index) * test_size)
    data = data.iloc[p].reset_index(drop=True) #シャッフル

    test_data = data[:testsize]
    train_data = data[testsize:]
    return train_data, test_data
