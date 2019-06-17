import arff
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

def read_sample_data(path):
    
    data = arff.load(open(path))
    cnames = [i[0] for i in data['attributes']]
    df = pd.DataFrame(data['data'], columns=cnames)
    df.sample(7500)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, train_size=5000)
    return X_tr, X_ts, y_tr, y_ts

def random_gridsearch(estimator, grid_dict):
    grid_search = RandomizedSearchCV(estimator=estimator, param_distributions=grid_dict,
        n_iter=300, cv=3, verbose=5, random_state=42, n_jobs=-1)
    return grid_search
