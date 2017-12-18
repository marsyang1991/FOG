import sklearn.cluster as cluster
import pandas as pd
import numpy as np
import utils

if __name__ == "__main__":
    feature_1 = pd.read_csv('./features/S01R02.txt', index_col=0)
    fogs = utils.find_fog(feature_1.iloc[:, -1])
    XX = []
    n = 10
    y_true = []
    for item in fogs:
        cur = np.array(feature_1.iloc[item.start_fog-n:item.end_fog+n, :-1]).tolist()
        XX.extend(cur)
        y_true.extend(np.array(feature_1.iloc[item.start_fog-n:item.end_fog+n, -1]).tolist())
    y_pred = cluster.KMeans(n_clusters=2,n_init=10).fit_predict(XX)
    print(len(XX))
    print(len(y_pred))
    for i in range(0,len(y_pred)):
        print('%d%s%d%s%d'%(i,'-->pred:',y_pred[i],',true:',y_true[i]))
