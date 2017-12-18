import numpy as np
import pandas as pd

from useless import preprocess_data as pre, Sliding_window as sw


class Config(object):
    def __init__(self):
        pass


def load_dataset(filename):
    data = pd.read_csv(filename, header=None)
    data_x, data_y = pre.process_dataset_file(np.array(data))
    data_x = sw.sliding_window(data_x, (64, data_x.shape[1]), (32, data_x.shape[1]))
    data_y = np.asarray([i[-1] for i in sw.sliding_window(data_y, 64, 32)])
    return data_x, data_y

if __name__=="__main__":
    filename = "data_seq/S01R010.txt"
    data_x,data_y = load_dataset(filename=filename)
    print data_x.shape, data_y.shape
