    
import pandas as pd
import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, f1_score, accuracy_score

def import_arff(path):
    data = arff.load(open(path))
    cnames = [i[0] for i in data['attributes']]
    df = pd.DataFrame(data['data'], columns=cnames)
    return df

def test3(estimator, df):    
    sizes = [0.5, 0.4, 0.3, 0.2, 0.1]

    dict_ = {'acc': [], 
             'precision': [],
             'recall': [], 
             'kappa': [], 
             'f1': []
            }

    for i, size in enumerate(sizes, start=1):
        train_X, test_X, train_y, test_y = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], shuffle=False, test_size=size)

        estimator.fit(train_X, train_y)
        pred = estimator.predict(test_X)

        dict_['acc'].append(accuracy_score(test_y, pred))
        dict_['precision'].append(precision_score(test_y, pred, average='weighted'))
        dict_['recall'].append(recall_score(test_y, pred, average='weighted'))
        dict_['kappa'].append(cohen_kappa_score(test_y, pred))
        dict_['f1'].append(f1_score(test_y, pred, average='weighted'))

        estimator_name = estimator.__class__.__name__

        print('{} of {} is completed for {}'.format(i, len(sizes), estimator_name))

    results = pd.DataFrame(dict_, index=[[estimator_name]*len(sizes), [1-i for i in sizes]])
    return results