import configparser
from utils.layer import Layer
from utils.loss import LossCalculator
from utils.earlystopping import EarlyStop
import numpy as np
import pickle
import matplotlib.pyplot as plt


class MultilayerPerception():
    """MultilayerPerception class"""
    def __init__(self, cfg="./config/setting.cfg", load_config=True, LR=0.1, Epochs=100, Batch=8, Bias=1):
        self.config = configparser.ConfigParser()
        self.config.read(cfg)
        self._label = ["M", "B"]
        self._loss = []  #訓練データを用いた損失関数の結果
        self._val_loss = []  #検証データを用いた損失関数の結果
        self.train_acc = []  #訓練データを用いた精度測定
        self.val_acc = []  #検証データを用いた精度測定
        self.F1 = []  #F1値の測定
        # 設定ファイルの読み込み
        if load_config is True:
            self.load_config()
        else:
            self.LR=LR
            self.Epochs=Epochs
            self.Batch=Batch
            self.Bias=Bias
        # EalryStopの初期化
        self.earlystop = self.init_earlystopping()


    def init_earlystopping(self):
        # 設定ファイル上にEarlyStopがあり、有効ならクラスを返す
        if self.config['ES']['earlystop'] and self.config['ES']['earlystop'] == "on":
            return EarlyStop(float(self.config['ES']['loss_border']), int(self.config['ES']['patience']))
        else:
            return None


    def load_config(self):
        self.LR = float(self.config['env']['LEARNING_RATE'])
        self.Epochs = int(self.config['env']['EPOCH'])
        self.Batch = int(self.config['env']['BATCH'])
        self.Bias = int(self.config['env']['BIAS'])


    def init_network(self, data):
        #設定ファイルからレイヤーの数を読み込む
        self.layer_num = len(self.config['network'])
        #ネットワークとウェイトを初期化
        self.network = []
        self.weight = []
        #最初の入力値の数（使用する特徴量の数）。本来は特徴量+バイアス項分あるが、それは後で追加する
        input = data.shape[1] - 1
        #層の数だけnetworkにlayerクラスを追加する
        for i in range(1, self.layer_num + 1):
            s = "layer" + str(i)
            output, activation = self.config['network'][s].split(',')
            output = int(output)
            weight = np.random.randn(input + 1, output) * 0.01
            self.weight.append(weight)
            layer = Layer(input + 1, output, weight, activation)
            self.network.append(layer)
            input = output


    def fit(self, data, test):
        """train model
        :param data : dataset for training
        :param test : dataset for test and calculate accuracy"""
        #テストデータを特徴量のデータと解答データに分ける
        testX = test.drop(columns="Diagnosis")
        testY = test["Diagnosis"]
        #テストの解答データをバイナリに変換　[n * 1]の配列（Mなら1, Bなら0）
        test_ans = np.where(testY == 'M', 1, 0)
        #テストの解答データを二次元に変換 [n * 2]の配列（Mなら[1, 0], Bなら[0,1]）
        test_vectorY = np.zeros((len(testY), len(self._label)))
        for i in range(len(testY)):
            test_vectorY[i, self._label.index(testY[i])] = 1

        #損失関数の初期化
        loss = LossCalculator(self.config['loss']['calculator'])

        for epoch in range(self.Epochs):
            step = 0.0000001 #学習率を徐々に下げて安定させるステップの設定
            # 訓練データの順番をシャッフルする（確率的勾配降下法）
            newData = data.sample(frac=1).reset_index(drop=True)
            #　シャッフルしたデータを、特徴量の行列と解答の行列データに分ける
            x = newData.drop(columns='Diagnosis')
            y = newData['Diagnosis']
            # 解答データを二次元に変換
            vectorY = np.zeros((len(y), len(self._label)))
            for i in range(len(y)):
                vectorY[i, self._label.index(y[i])] = 1
            #エポック毎に損失を計測するが、それはバッチ毎に計算した平均を取る
            errors = []
            #バッチ毎に計算を行う
            for i in range(data.shape[0] // self.Batch):
                #バッチ毎にスタート地点と終了地点を設定
                start = i * self.Batch
                end = start + self.Batch
                #バッチデータで予測をたてる
                prediction = self.forward(x[start:end], vectorY[start:end])
                #予測からバックプロパゲーションを用いた勾配更新
                self.backward(prediction, vectorY[start:end])
                # 予測値から損失関数を計算し、errorsに入れる
                error_mean = loss.func(prediction, vectorY[start:end])
                errors.append(error_mean)

            #検証データのチェック
            val_prediction = self.forward(testX, test_vectorY)
            val_loss = loss.func(val_prediction, test_vectorY)
            loss_mean = np.mean(errors)
            self._loss.append(loss_mean)
            self._val_loss.append(val_loss)

            #検証データのチェック2
            current_prediction = self.predict(x, binary=True)
            current_val_prediction = self.predict(testX, binary=True)
            train_ans = np.where(y == 'M', 1, 0)
            train_acc = np.mean(current_prediction == train_ans)
            val_acc = np.mean(current_val_prediction == test_ans)
            self.train_acc.append(train_acc)
            self.val_acc.append(val_acc)

            # 検証データのチェック3 分類モデルの性能を評価するための指標で、適合率 (Precision) と 再現率 (Recall) の調和平均を用います
            TP = sum((t == 1 and p == 1) for t, p in zip(train_ans, current_prediction))
            FP = sum((t == 0 and p == 1) for t, p in zip(train_ans, current_prediction))
            FN = sum((t == 1 and p == 0) for t, p in zip(train_ans, current_prediction))
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            self.F1.append(f1_score)

            print(f"epoch {epoch}/{self.Epochs} - loss: {loss_mean:.10f} - val_loss: {val_loss:.10f} - F1: {f1_score:.10f}")


            if self.earlystop is not None:
                if self.earlystop.evaluate(loss_mean, val_loss, self.weight):
                    self.weight = self.earlystop.recovery()
                    return
            self.LR -= step


    def forward(self, x, ans):
        """feed forwarding"""
        #レイヤー毎に前に計算を進めていく
        x = np.array(x)
        for layer in self.network:
            x = layer.forward(x, self.Bias)
        prediction = x
        return prediction

    def backward(self, y_pred, y_true):
        """back propagation"""
        #レイヤー毎に後ろに勾配計算を勧めていく
        gradient = y_pred - y_true
        for i, layer in enumerate(reversed(self.network)):
            gradient = layer.backward(gradient, self.LR)
            self.weight[len(self.network) - i - 1] = layer.weight

    def predict(self, data, binary=False):
        """
        学習済みモデルを使ってデータを予測する
        :param data: 入力データ (特徴量のみ)
        :param binary: binary == Trueのとき、予測値を1, 0で返す
        :return: クラスラベルまたは予測値
        """
        #出力の予測
        x = np.array(data)
        for layer in self.network:
            x = layer.forward(x, self.Bias)

        predictions = (x[:, 0] > 0.5).astype(int)
        if binary == True:
            return predictions
        label_map = {0: 'B', 1: 'M'}
        return [label_map[p] for p in predictions]

    def save_model(self, filename="./saved_model.pkl"):
        """save model in the type of .pkl
        param filename: (str)"""
        print(f"> saving model '{filename}'...")
        model_data = {
            "weights": self.weight,
            "network": self.network
        }
        with open(filename, "wb") as f:
            pickle.dump(model_data, f)


    def load_model(self, filename="./saved_model.pkl"):
        """load model from the type of .pkl
        param filename: (str)"""
        print(f"> loading model '{filename}'...")
        with open(filename, "rb") as f:
            model_data = pickle.load(f)
        self.weight = model_data["weights"]
        self.network = model_data["network"]


    def create_loss_graphs(self):
        """create graphs which labels of x, y are loss and Epochs"""
        epochs = list(range(1, len(self._loss) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self._loss, label="training loss", linestyle="--")
        plt.plot(epochs, self._val_loss, label="validation loss")
        plt.title("Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)


    def create_acc_graphs(self):
        """create graphs which labels of x, y are accuracy and Epochs"""
        epochs = list(range(1, len(self.train_acc) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_acc, label="training acc")
        plt.plot(epochs, self.val_acc, label="validation acc")
        plt.title("Acc per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Accurancy")
        plt.legend()
        plt.grid(True)


    def create_F1_graphs(self):
        """create graphs which labels of x, y are F1 score and Epochs"""
        epochs = list(range(1, len(self.F1) + 1))
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_acc, label="F1 score")
        plt.title("F1 per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("F1 score")
        plt.legend()
        plt.grid(True)


    def show_graphs(self):
        """show graphs which are created already"""
        plt.show()








