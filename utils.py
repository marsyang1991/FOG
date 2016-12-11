# coding = utf-8
import numpy as np
from scipy.stats import mode
import os
import pandas as pd
from model import FOG as fog
import random
import matplotlib.pyplot as plt
import hmmlearn as hmm

import numpy as np
import warnings
from sklearn import metrics

window_length = 4
step_length = 2


def sliding_window(data, window, overlap=0):
    window = int(window)
    overlap = int(overlap)
    length = len(data)
    start = 0
    end = start + window - 1
    items = []
    labels = []
    while end < length:
        if end >= length:
            break
        temp = np.transpose(np.array(data.loc[start:end, 'ankle_vertical']))
        # ta = mode(data.loc[start:end, 'annotation']).mode[0]
        if (np.array(data.loc[start:end, 'annotation']) == 2).any():
            ta = 2
        else:
            ta = 1
        # ta1 = data.loc[end, 'annotation']
        labels.append(ta)
        items.append(temp)
        start += window - overlap
        end = start + window - 1
    return labels, items


# extract features from data frame
def get_feature(frame):
    # print pd.DataFrame(frame).shape
    f_mean = np.mean(frame)
    d_mean = frame - np.mean(frame)
    d_fft = np.fft.fft(d_mean, 256)
    d_1 = abs(d_fft * np.conj(d_fft)) / 256
    # d_mode = np.abs(d_fft) / 256
    # power low
    PL = x_numericalIntegration(d_1[1:13], 64)
    # Dominant Frequency
    DF = np.argmax(d_1[1:33])
    # Dominant Frequency Amplitude
    DFA = np.max(d_1[1:33])

    PF = x_numericalIntegration(d_1[13:33], 64)
    if PL == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    TP = PL + PF
    FI = PF / PL
    mx = np.max(frame)
    mn = np.min(frame)
    sd = np.std(frame)
    var = np.var(frame)
    cov = sd / f_mean
    return [f_mean, sd, var, cov, mx, mn, PL, DF, DFA, PF, TP, FI]


# merge data in file_dir into one list
def get_all_data_from_dir(file_dir):
    list = os.listdir(file_dir)
    datas = []
    for file in list:
        data = pd.read_csv(os.path.join(file_dir, file))
        datas.append(data)
    return datas


# delete the annotation 0 and saved into the direction "ok_data" and keep the original file name
def deal_with_0():
    file_dir = './dataset_fog_release/dataset/'
    file_list = os.listdir(file_dir)
    for file in file_list:
        data = pd.read_csv(os.path.join(file_dir, file), delimiter=' ', header=None)
        data.columns = ['timestamp', 'ankle_hori_fw', 'ankle_vertical', 'ankle_hori_lat', 'thigh_hori_fw',
                        'thigh_vertical',
                        'thigh_hori_lat', 'trunk_hori_fw', 'trunk_vertical', 'trunk_hori_lat', 'annotation']
        temp = data[data["annotation"] != 0]
        pd.DataFrame(temp).to_csv('./ok_data/' + file, index=False)
        print('deal_with_0: {}, data length:{}'.format(file, temp.shape))


# find fog in raw data
def extract_fogs(data, file_name=""):
    start = 0
    in_fog = False
    fogs = []
    for index in range(0, data.shape[0]):
        label = data.loc[index, 'annotation']
        if label == 0:
            continue
        if label == 2 and not in_fog:
            start = index
            in_fog = True
        if label == 1 and in_fog:
            end = index
            new_fog = fog(start, end)
            fogs.append(new_fog)
            in_fog = False
    if len(fogs) < 1:
        return []
    for item in fogs:
        dd = data.iloc[item.start:item.end, :]
        item.set_data(dd)
        item.set_file_name(file_name)
    return fogs


# save fogs into a file
def save_fogs(fogs, file_name='./fogs.csv'):
    ll = []
    for item in fogs:
        temp = [item.file_name, item.start_fog, item.end_fog]
        ll.append(temp)
    ll_df = pd.DataFrame(ll)
    ll_df.columns = ['file_name', 'start', 'end']
    ll_df.to_csv(file_name)


# for the certain raw_data set, segment into frames and extracted features
def get_features_by_file(filename):
    print("get_features_by_file:" + filename)
    raw_data = pd.read_csv(os.path.join('./ok_data', filename))
    labels, frames = sliding_window(raw_data, window=64 * window_length, overlap=(window_length - step_length) * 64)
    features_with_label = []
    for index in range(0, len(frames)):
        feature = get_feature(frames[index])
        label = labels[index] - 1
        feature.append(label)
        features_with_label.append(feature)
    pd.DataFrame(features_with_label).to_csv(os.path.join('./features', filename))


def write_feature():
    file_list = os.listdir('./ok_data')
    for file in file_list:
        get_features_by_file(file)


def x_numericalIntegration(x, sr):
    return (np.sum(x[0:-1]) + np.sum(x[1:])) / (2 * sr)


def get_random_samples(X, y, count, n_frame):
    print("get random samples")
    X_result = []
    lengths_result = []
    for index in range(0, count):
        i = random.randint(0, len(y) - 1)
        while y[i] > 0 or i < n_frame:
            i = random.randint(0, len(y) - 1)
        X_result.extend(np.array(X[i - n_frame:i, :]).tolist())
        lengths_result.append(n_frame)
    return X_result, lengths_result


def see_result_by_img(y_true, y_pred):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
    xx = range(0, len(y_pred))
    ax1.plot(xx, y_true, 'r*')
    ax1.set_title("true")
    ax2.plot(xx, y_pred, 'b*')
    ax2.set_title("predict")
    plt.show()


def load_feature_by_user():
    print("load_feature_by_user,return a list whose length 10")
    # 1
    feature_1 = pd.read_csv('./features/S01R01.txt', index_col=0)
    feature_2 = pd.read_csv('./features/S01R02.txt', index_col=0)
    feature_s01 = feature_1.append(feature_2)
    # 2
    feature_s02_1 = pd.read_csv('./features/S02R01.txt', index_col=0)
    feature_s02_2 = pd.read_csv('./features/S02R02.txt', index_col=0)
    feature_s02 = feature_s02_1.append(feature_s02_2)
    # 3
    feature_s03_1 = pd.read_csv('./features/S03R01.txt', index_col=0)
    feature_s03_2 = pd.read_csv('./features/S03R02.txt', index_col=0)
    feature_s03_3 = pd.read_csv('./features/S03R03.txt', index_col=0)
    feature_s03 = feature_s03_1.append(feature_s03_2).append(feature_s03_3)
    # 4
    feature_s04 = pd.read_csv('./features/S04R01.txt', index_col=0)
    # 5
    feature_s05_1 = pd.read_csv('./features/S05R01.txt', index_col=0)
    feature_s05_2 = pd.read_csv('./features/S05R02.txt', index_col=0)
    feature_s05 = feature_s05_1.append(feature_s05_2)
    # 6
    feature_s06_1 = pd.read_csv('./features/S06R01.txt', index_col=0)
    feature_s06_2 = pd.read_csv('./features/S06R02.txt', index_col=0)
    feature_s06 = feature_s06_1.append(feature_s06_2)
    # 7
    feature_s07_1 = pd.read_csv('./features/S07R01.txt', index_col=0)
    feature_s07_2 = pd.read_csv('./features/S07R02.txt', index_col=0)
    feature_s07 = feature_s07_1.append(feature_s07_2)
    # 8, 9, 10
    feature_s08 = pd.read_csv('./features/S08R01.txt', index_col=0)
    feature_s09 = pd.read_csv('./features/S09R01.txt', index_col=0)
    feature_s10 = pd.read_csv('./features/S10R01.txt', index_col=0)
    return [feature_s01, feature_s02, feature_s03, feature_s04, feature_s05, feature_s06, feature_s07, feature_s08,
            feature_s09, feature_s10]


def find_fog(seq):
    diff = np.diff(seq)
    diff_index = (diff != 0)
    return len(diff[diff_index]) / 2


def print_result(cm):
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    print cm
    print("Specifity={}\tSensitivity={}".format(float(TN) / (TN + FP), float(TP) / (TP + FN)))


def eva_gmmhmm(X_train, y_train, X_test, y_test, n_component=2, n_mix=10, n_sequence=6):
    X_all = X_train
    y_all = y_train
    X_hmm_1 = []
    X_hmm_0 = []
    lengths_hmm_0 = []
    lengths_hmm_1 = []
    n = n_sequence
    for index in range(n_sequence, len(y_all)):
        cur = np.array(X_all[index - n:index, :]).tolist()
        if y_all[index] == 0:
            X_hmm_0.extend(cur)
            lengths_hmm_1.append(0)
        else:
            X_hmm_1.extend(cur)
            lengths_hmm_1.append(n)
    # X_hmm_0, lengths_hmm_0 = get_random_samples(X_all, y_all, len(lengths_hmm), n)
    hmm_0 = hmm.GMMHMM(n_components=n_component, n_mix=n_mix)
    hmm_0.fit(X=X_hmm_0, lengths=lengths_hmm_0)
    hmm_1 = hmm.GMMHMM(n_components=n_component, n_mix=n_mix)
    hmm_1.fit(X=X_hmm_1, lengths=lengths_hmm_1)
    y_true = []
    y_pred = []
    for index in range(0, X_test.shape[0]):
        if index <= n:
            sample = X_test[:index + 1, :]
        else:
            sample = X_test[index - n:index, :]
        y_true.append(y_test[index])
        result_1 = hmm_1.score(sample)
        result_0 = hmm_0.score(sample)
        if result_0 > result_1:
            y_pred.append(0)
        else:
            y_pred.append(1)
    cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
    see_result_by_img(y_true, y_pred)
    print_result(cm)
    print("true: {} fogs".format(find_fog(y_true)))
    print("pred: {} fogs".format(find_fog(y_pred)))
    return cm


if __name__ == "__main__":
    # write_feature()
    warnings.filterwarnings("ignore")
    features = load_feature_by_user()
    feature_train = pd.DataFrame(features[1]).append(pd.DataFrame(features[0]))
    feature_test = pd.DataFrame(features[2])
    X_train = np.array(feature_train.iloc[:, :-1])
    y_train = np.array(feature_train.iloc[:, -1])
    X_test = np.array(feature_test.iloc[:, :-1])
    y_test = np.array(feature_test.iloc[:, -1])
    eva_gmmhmm(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
