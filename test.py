import pandas as pd
import numpy as np
import os
from model import FOG as fog
from utils import sliding_window, get_all_data_from_dir, get_feature,write_feature
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
import random
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

if __name__ == "__main__":
    freq = 64
    window_time = 4
    step_time = 0.5
    datas = get_all_data_from_dir('./ok_data')
    all_labels = []
    all_features = []
    for data in datas:
        # segment into frames
        labels, frames = sliding_window(data=data, window=freq * window_time, overlap=freq * (window_time - step_time))
        # extract features
        for i in range(0, len(frames)):
            feature = get_feature(frame=frames[i])
            if feature == -1:
                continue
            all_labels.append(labels[i])
            all_features.append(feature)
    clf = tree.DecisionTreeClassifier()

    X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=0.4, random_state=42)
    clf.fit(X=X_train, y=y_train)
    y_predicted = clf.predict(X=X_test)
    print metrics.confusion_matrix(y_test,y_pred=y_predicted)
    print metrics.accuracy_score(y_test,y_predicted)
    print metrics.recall_score(y_test,y_predicted)