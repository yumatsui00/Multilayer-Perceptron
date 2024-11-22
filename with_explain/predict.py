import configparser
import pandas as pd
import utils.utils as ut
from utils.model import MultilayerPerception
import sys
import os

def data_preprocess_for_predict(data):
    """preprocess Dataset. Extract useful data columns, standarize
        arguments :
            `data` : (DataFrame)
                the path where we will save the csv file (relative or absolute path)
    """
    valid_columns = config['columns']['column_names'].split(',')
    bad1 = config['columns']['bad_columns1'].split(',')
    bad2 = config['columns']['bad_columns2'].split(',')
    bad3 = config['columns']['bad_columns3'].split(',')

    bads = bad1 + bad2 + bad3
    data.columns = valid_columns
    data = data.drop(columns='ID').drop(columns=bads).drop(columns="Diagnosis")
    data = ut.standarization(data)
    return data



def multilayer_predict(data):
    data = data_preprocess_for_predict(data)
    model = MultilayerPerception()
    model.load_model()
    predictions = model.predict(data)
    print(predictions)



if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise AssertionError("Invalid Args")
        if not os.path.exists(sys.argv[1]):
            raise AssertionError(f"{sys.argv[1]} doesn't exist")
        elif not os.access(sys.argv[1], os.R_OK):
            raise AssertionError("Permission Denied")
        data = pd.read_csv(sys.argv[1])
        config = configparser.ConfigParser()
        config.read("./config/setting.cfg")
        multilayer_predict(data)

    except AssertionError as e:
        print(f"Error: {e}")
