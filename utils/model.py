import configparser
from utils.layer import Layer
import numpy as np

#!バイアス項の処理ができてないと思う
class MultilayerPerception():
    """MultilayerPerception class"""
    def __init__(self, load_config=True, LR=0.1, Epochs=100, Batch=8, Iter=10):
        self.config = configparser.ConfigParser()
        self.config.read("./config/setting.cfg")
        self._label = ["M", "B"]
        if load_config is True:
            self.load_config()
        else:
            self.LR=LR
            self.Epochs=Epochs
            self.Batch=Batch
            self.Iter=Iter


    def load_config(self):
        self.LR = float(self.config['env']['LEARNING_RATE'])
        self.Epochs = int(self.config['env']['EPOCH'])
        self.Batch = int(self.config['env']['BATCH'])
        self.Iter = int(self.config['env']['ITER'])


    def init_network(self, data):
        self.layer_num = len(self.config['network'])
        self.network = []
        self.weight = []  #weight の初期化も個々でおこナウ
        input = data.shape[1] - 1  #特徴量の数が最初のinputになる
        #self.network.append(Layer(input, input, "Input"))
        #ひとまず、ニューロンの間を一つの層として数える。ニューロン自体ではなく
        for i in range(1, self.layer_num + 1):
            s = "layer" + str(i)
            output, activation = self.config['network'][s].split(',')
            output = int(output)
            #!バイアス項の分だけ＋１してる
            layer = Layer(input + 1, output, activation)
            self.network.append(layer)
            self.weight.append(np.zeros((input + 1, output)))
            input = output


    def load_weight(self, weight_path):
        """load weight here"""
        if weight_path:
            #TODO pathをloadする
            print()



    def fit(self, data):
        """train self here"""
        #weightの初期化済み
        #data の標準化済

        for epoch in range(self.Epochs):
            #データをランダムにして、正解とデータに分ける
            newData = data.sample(frac=1).reset_index(drop=True)
            x = newData.drop(columns='Diagnosis')
            y = newData['Diagnosis']
            #正解ラベルの初期化。Mは１列目が１、bは二列目が１
            vectorY = np.zeros((len(y), len(self._label)))
            for i in range(len(y)):
                vectorY[i, self._label.index(y[i])] = 1
            #?xの前処理も必要？必要そうなこと
            #バイアス項の追加、
            #TODO バッチごとにデータを取得して勾配を計算→更新
        #TODO 更新が終わったら重みを保存



