import configparser
from icecream import ic
import pandas as pd
import numpy as np
import utils.utils as ut
from utils.model import MultilayerPerception


def data_preprocess(data: pd.DataFrame) -> pd.DataFrame:
    valid_columns = config['columns']['column_names'].split(',')
    bad1 = config['columns']['bad_columns1'].split(',')
    bad2 = config['columns']['bad_columns2'].split(',')
    bad3 = config['columns']['bad_columns3'].split(',')
    bad4 = config['columns']['bad_columns4'].split(',')
    bad5 = config['columns']['bad_columns5'].split(',')
    bad6 = config['columns']['bad_columns6'].split(',')
    bads = bad1 + bad2 + bad3 + bad4 + bad5 + bad6
    data.columns = valid_columns
    data = data.drop(columns='ID').drop(columns=bads)
    #dataY = data['Diagnosis']
    #dataX = data.drop(columns="Diagnosis")
    data = ut.standarization(data)
    train, test= ut.split_train_test(data, float(config['env']['TESTSIZE']), int(config['env']['SEED']))
    train = train.reset_index().drop(columns="index")
    return data, train, test


def multilayer_training(data):
    data, train, test = data_preprocess(data)
    ic(train)
    model = MultilayerPerception()
    model.init_network(data)
    #ic(model.weight)
    #ic(model.network[4].__dict__)
    model.fit(train)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config/setting.cfg")
    data = pd.read_csv("./data/data.csv")
    multilayer_training(data)
