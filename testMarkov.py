from hmmlearn import hmm
import utils
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    file_path = "~/Documents/pyTest/HMMELM/ok_data/S01R01.txt"
    raw_data = utils.read_data_from(file_path=file_path)
    print("Have read data:", raw_data.shape)
    # segmentation
    fs = 64
    window_time = 2
    overlap_time = 0
    frames = utils.sliding_window(raw_data, window=fs * window_time, overlap=fs * overlap_time)
    # extract features
    features = []
    for frame in frames:
        feature = utils.get_feature(frame=frame, col_range=[1, 2, 3])
        features.append(feature)
    features = np.array(features)
    pre_length = 4
    split_data = utils.abandon_pre_fog(features[:, -1], pre_length)

    # train GMM-HMM
    frame_num = 4
    fog_indices = split_data['fog_indices']
    pre_fog_indices = split_data['pre_fog_indices']
    normal_indices = split_data['normal_indices']
    normal_down_indices = utils.down_sampling(normal_indices, 2 * len(fog_indices))
    fog_train, fog_test = train_test_split(fog_indices, test_size=0.1, random_state=0)
    normal_train, normal_test = train_test_split(normal_down_indices, test_size=0.1, random_state=0)
    data = utils.get_sequence_data(features, fog_train, frame_num, (window_time - overlap_time) * 1000 * 2)
    data_1 = utils.get_sequence_data(features, normal_train, frame_num, (window_time - overlap_time) * 1000 * 2)
    print('data for fog:', np.array(data['lengths']).shape[0], np.array(data_1['lengths']).shape[0])
    data_test = utils.get_sequence_data(features, fog_test, frame_num, (window_time - overlap_time) * 1000 * 2)
    data1_test = utils.get_sequence_data(features, normal_test, frame_num, (window_time - overlap_time) * 1000 * 2)
    print('data test for fog:', np.array(data_test['lengths']).shape[0], np.array(data1_test['lengths']).shape[0])
    hmm_0 = hmm.GMMHMM(n_components=10, n_mix=3)
    hmm_0.fit(X=data['data'], lengths=data['lengths'])
    hmm_1 = hmm.GMMHMM(n_components=10, n_mix=3)
    hmm_1.fit(X=data_1['data'], lengths=data['lengths'])
    acc = 0
    m_class = np.zeros(len(data_test['lengths']))
    for i in range(len(data_test['lengths']) - 1):
        try:
            score0 = hmm_0.score(np.array(data_test['data'])[i * frame_num:(i + 1) * frame_num - 1, :])
            score1 = hmm_1.score(np.array(data_test['data'])[i * frame_num:(i + 1) * frame_num - 1, :])
            if score0 > score1:
                m_class[i] = 2
            else:
                m_class[i] = 1
            if m_class[i] == 2:
                acc += 1
        except Exception as e:
            print(e)
    acc1 = 0
    m_class1 = np.zeros(len(data1_test['lengths']))
    for i in range(len(data1_test['lengths']) - 1):
        score0 = hmm_0.score(np.array(data1_test['data'])[i * frame_num:(i + 1) * frame_num - 1, :])
        score1 = hmm_1.score(np.array(data1_test['data'])[i * frame_num:(i + 1) * frame_num - 1, :])

        if score0 > score1:
            m_class1[i] = 2
        else:
            m_class1[i] = 1
        if m_class1[i] == 1:
            acc1 += 1
    print('fog:{0}/{1}, normal:{2}/{3}'.format(acc, len(m_class), acc1, len(m_class1)))

    data_all = utils.get_sequence_data(features, range(len(features)), frame_num,
                                       (window_time - overlap_time) * 1000 * 2)
    score = []
    for i in range(len(data_all['lengths']) - 1):
        score.append(hmm_0.score(np.array(data_all['data'])[i * frame_num:(i + 1) * frame_num - 1, :]))
    plt.plot(score)
    plt.show()