import numpy as np


def cross_entropy(y_pred, y_true):
    """
    クロスエントロピー損失を計算する
    :param y_pred: モデルの予測値 (softmax出力), shape=(batch_size, num_calsses(2))
    :param y_true: 正解ラベル (ワンホットエンコード), shape=(batch_size, num_classes(2))
    :return: 平均損失 (スカラー値)
    """
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    loss_matrix = -np.sum(y_true * np.log(y_pred), axis=1)
    return loss_matrix.mean()




class LossCalculator():
	""""Loss class"""
	def __init__(self, types):
		if types == "cross-entropy":
			self.func = cross_entropy


