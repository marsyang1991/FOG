from utils import sliding_window, get_all_data_from_dir, get_feature, get_random_samples
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn import hmm
import numpy as np
import warnings
from sklearn import metrics
warnings.filterwarnings("ignore")

feature_1 = pd.read_csv('./features/S01R01.txt', index_col=0)
feature_2 = pd.read_csv('./features/S01R02.txt', index_col=0)
feature_S01 = feature_1.append(feature_2)
feature_S02_1 = pd.read_csv('./features/S02R01.txt', index_col=0)
feature_S02_2 = pd.read_csv('./features/S02R02.txt', index_col=0)
feature_S02 = feature_S02_1.append(feature_S02_2)
feature_S03_1 = pd.read_csv('./features/S03R01.txt', index_col=0)
feature_S03_2 = pd.read_csv('./features/S03R02.txt', index_col=0)
feature_S03_3 = pd.read_csv('./features/S03R03.txt', index_col=0)
feature_S03 = feature_S03_1.append(feature_S03_2).append(feature_S03_3)
feature_train = feature_S02.append(feature_S01)
feature_test = feature_S03
X_train = np.array(feature_train.iloc[:, :-1])
y_train = np.array(feature_train.iloc[:, -1])
X_test = np.array(feature_test.iloc[:, :-1])
y_test = np.array(feature_test.iloc[:, -1])
y_all = y_train
X_all = X_train
X_hmm = []
lengths_hmm = []
for index in range(0, len(y_all)):
    if y_all[index] == 0:
        continue
    else:
        cur = np.array(X_all[index - 10:index, :]).tolist()
        X_hmm.extend(cur)
        lengths_hmm.append(10)

X_hmm_0, lengths_hmm_0 = get_random_samples(X_all, y_all, len(lengths_hmm))
hmm_0 = hmm.GMMHMM(n_components=4, n_mix=10)
hmm_0.fit(X=X_hmm_0, lengths=lengths_hmm_0)
hmm_1 = hmm.GMMHMM(n_components=4, n_mix=10)

hmm_1.fit(X=X_hmm, lengths=lengths_hmm)

y_true = []
y_pred = []
for index in range(0, X_test.shape[0]):
    if index <= 10:
        sample = X_test[:index + 1, :]
    else:
        sample = X_test[index - 10:index, :]
    y_true.append(y_test[index])
    result_1 = hmm_1.score(sample)
    result_0 = hmm_0.score(sample)
    if result_0 > result_1:
        y_pred.append(0)
    else:
        y_pred.append(1)
cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)
xx = range(0, len(y_pred))
ax1.plot(xx, y_true, 'r*')
ax1.set_title("true")
ax2.plot(xx, y_pred, 'b*')
ax2.set_title("predicted")
plt.show()
print(cm)