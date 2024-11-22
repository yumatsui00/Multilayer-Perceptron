import numpy as np


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)  # オーバーフロー防止
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax_prime(z): #softmax はとりあえず最初にやってる
    return 1


activation_functions = {
    "sigmoid": sigmoid,
    "softmax": softmax
}

prime_functions = {
    "sigmoid": sigmoid_prime,
    "softmax": softmax_prime
}

class Layer():
    """Layer class"""
    def __init__(self, input, output, weight, activation):
        """params:
            input : (int)
                the number of input of the layer
            output : (int)
                the number of output of the layer
            weight : (array)
                current weight of the layer
            activation : (str)
                kinds of activation_functions"""
        #入力数と出力数と重みの初期化
        self.input = input
        self.output = output
        self.weight = weight
        # 活性化関数の設定
        if activation in activation_functions:
            self.activation = activation_functions[activation]
            self.activation_prime = prime_functions[activation]
        else:
            raise AssertionError("Invalid argument for activation in Layer")

    def forward(self, x, bias):
        """add bias and, feed forward using current weight
        params :
            x : (array)
                values before the layer
            bias : (int)
                value of bias to add
            ret(x) : (array)
                values after the layer"""
        #バイアス項を追加（バイアス固定）
        x = np.hstack([[[bias] for _ in range(x.shape[0])], x])
        #バイアス付きが入力の行列になる
        self.input_matrix = x
        #重みとの内積を取って、活性化関数を通すことで出力を計算
        x = x.dot(self.weight)
        self.output_matrix = self.activation(x)
        return self.output_matrix

    def backward(self, gradient_output, LearningRate):
        """calculate gradient backwards and update weights
        param :
            gradient_output : (array)
                gradient which comes from the right side
            LearningRate : (float)
                learning rate to update weight
            ret(gradient_input) : (array)
                gradient which go out of left side"""
        #活性化関数の微分に出力を通して勾配とかけることで、修正された勾配を計算
        gradient = gradient_output * self.activation_prime(self.output_matrix)
        #重みの更新値を計算。入力行列と修正後の勾配の内積
        weight = self.input_matrix.T.dot(gradient)
        #次にでんぱするべき勾配を計算。ただしバイアス項は除外
        gradient_input = gradient.dot(self.weight.T)
        gradient_input = gradient_input[:, 1:]
        #実際の重みの更新
        self.weight -= LearningRate * weight
        return gradient_input

