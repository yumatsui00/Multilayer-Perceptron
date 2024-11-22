import configparser
import pandas as pd
import numpy as np
import utils.utils as ut
from utils.model import MultilayerPerception


def data_preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """preprocess Dataset. Extract useful data columns, standarize them, and split it into train and test data
        arguments :
            `data` : (DataFrame)
                dataset
    """
    # データから、不要な（相関のなさそうな）情報を消す
    valid_columns = config['columns']['column_names'].split(',')
    bad1 = config['columns']['bad_columns1'].split(',')
    bad2 = config['columns']['bad_columns2'].split(',')
    bad3 = config['columns']['bad_columns3'].split(',')
    bads = bad1 + bad2 + bad3
    data.columns = valid_columns
    # IDもいらないので消す
    data = data.drop(columns='ID').drop(columns=bads)

    #標準化
    data = ut.standarization(data)
    # データを訓練データと￥検証データに分ける
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
    #データの前処理
    data, train, test = data_preprocess(data)

    model = MultilayerPerception(cfg=cfg)
    #多層パーセプトロンのネットワーク初期化
    model.init_network(data)
    # 訓練
    model.fit(train, test)

    #訓練時に収集したデータからグラフ作成
    model.create_loss_graphs()
    model.create_acc_graphs()
    model.create_F1_graphs()
    model.show_graphs()

    #最終予測時に精度を検証するための解答作成
    ans = np.array(test['Diagnosis'])
    ans = np.where(ans == 'M', 1, 0)

    #最終予測
    pre = model.predict(test.drop(columns="Diagnosis"), binary=True)
    #最終予測とその精度測定
    accuracy = np.mean(pre == ans)
    print(f"Accuracy: {accuracy:.2%}")

    #モデルのセーブ
    model.save_model()


if __name__ == "__main__":
    #設定ファイル
    config = configparser.ConfigParser()
    config.read("./config/setting.cfg")
    data = pd.read_csv("./data_training.csv")
    multilayer_training(data)
