import pandas as pd






def print_all_data(data):
    pd.set_option('display.max_rows', None)                  #行の省略をなくす
    pd.set_option('display.max_columns', None)               #列の省略をなくす
    pd.set_option('display.expand_frame_repr', None)         #横長の出力の省略をなくす
    #print(data)
    print(data.describe())


if __name__ == "__main__":
    data = pd.read_csv("./data/data.csv")
    num_columns = data.shape[1]
    column_names = ["ID", "Diagnosis"] + [f"{stat}_{feature}" for stat in ["Mean", "Std", "Max"] for feature in [
    "Radius", "Texture", "Perimeter", "Area", "Smoothness",
    "Compactness", "Concavity", "ConcavePoints", "Symmetry", "FractalDimension"
    ]]
    data.columns = column_names
    #print_all_data(data)
    print(data.describe())
