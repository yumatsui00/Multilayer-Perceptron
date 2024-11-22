import configparser
import pandas as pd
import numpy as np
import utils.utils as ut
from utils.model import MultilayerPerception
import sys


def data_preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """preprocess Dataset. Extract useful data columns, standarize them, and split it into train and test data
        arguments :
            `data` : (DataFrame)
                dataset
    """
    valid_columns = config['columns']['column_names'].split(',')
    bad1 = config['columns']['bad_columns1'].split(',')
    bad2 = config['columns']['bad_columns2'].split(',')
    bad3 = config['columns']['bad_columns3'].split(',')

    bads = bad1 + bad2 + bad3
    data.columns = valid_columns
    data = data.drop(columns='ID').drop(columns=bads)

    data = ut.standarization(data)
    train, test= ut.split_train_test(data, float(config['env']['TESTSIZE']), int(config['env']['SEED']))
    train = train.reset_index().drop(columns="index")
    return data, train, test


def multilayer_training(data, cfg="./config/setting.cfg"):
    """train model and create graphs about training
        arguments :
            'data' : (DataFrame)
                datasets
            'cfg' : (str)
                path to config file to import
    """
    data, train, test = data_preprocess(data)
    model = MultilayerPerception(cfg=cfg)
    model.init_network(data)
    model.fit(train, test)

    ans = np.array(test['Diagnosis'])
    ans = np.where(ans == 'M', 1, 0)

    pre = model.predict(test.drop(columns="Diagnosis"), binary=True)

    model.create_loss_graphs()
    model.create_acc_graphs()
    model.create_F1_graphs()
    model.show_graphs()

    model.save_model()

    accuracy = np.mean(pre == ans)
    print(f"Accuracy: {accuracy:.2%}")



if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config/setting.cfg")
    if len(sys.argv) == 3:
        data = pd.read_csv(sys.argv[2])
    else:
        data = pd.read_csv("./data_training.csv")
    multilayer_training(data)
