import time
import pandas as pd

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import f1_score, recall_score, precision_score

elemination_type = 'original'  # oneToOne - oneToTwo - original
data_metrics_arr = []

data_list = ['all',
             'Android-Universal-Image-Loader',
             'antlr4',
             'elasticsearch',
             'junit',
             'MapDB',
             'mcMMO',
             'mct',
             'neo4j',
             'netty',
             'orientdb',
             'oryx',
             'titan',
             'ceylon-ide-eclipse',
             'hazelcast',
             'BroadleafCommerce'
             ]

for data_name in data_list:

    time_start = time.time()

    data = pd.read_csv(elemination_type+"_elimination_metrics/" +
                       elemination_type+"_elimination_metrics_"+data_name+".csv")

    X = data.iloc[:, 0:10].values
    Y = data.iloc[:, 10:11].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.4, random_state=10)

    xgbModel = XGBClassifier()

    xgbModel.fit(X_train, Y_train)
    Y_pred = xgbModel.predict(X_test)

    f_score = metrics.f1_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    prediction = precision_score(Y_test, Y_pred)
    accuracy = metrics.accuracy_score(Y_test, Y_pred)

    print('Data Name: ', data_name)
    print('F-Measure: %.3f' % f_score)
    print('Recall: %.3f' % recall)
    print('Prediction: %.3f' % prediction)
    print('Accuracy: %.3f' % accuracy)
    time_end = time.time()
    total_time = time_end-time_start
    print('Time: ', total_time)

    metrics_obj = {
        "Data Name": data_name,
        "Time": total_time,
        "F-Measure": f_score,
        "Recall": recall,
        "Prediction": prediction,
        "Accuracy": accuracy
    }

    data_metrics_arr.append(metrics_obj)


new_dataframe = pd.DataFrame(data_metrics_arr)
new_dataframe.to_excel("output_XGBClassifier_"+elemination_type+".xlsx")
