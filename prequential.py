from time import clock

import numpy as np
import pandas as pd
from tqdm import tnrange, tqdm_notebook

def prequential(n_init_tr, model, X, y, w):
    y_pred = []
    
    row_n = y.shape[0]

    for i in tnrange(row_n - n_init_tr):
        Xn = X.iloc[:n_init_tr+i,:]
        yn = y.iloc[:n_init_tr+i]
        if w>(n_init_tr+i):
            model.fit(Xn, yn)
            y_pred.append(model.predict(X.iloc[n_init_tr+i,:].values.reshape(1,-1)))
        else:
            model.fit(Xn.iloc[-w:,:], yn.iloc[-w:])
            y_pred.append(model.predict(X.iloc[n_init_tr+i,:].values.reshape(1,-1)))
        
    pred_match = np.equal(np.array(y_pred).reshape(1,-1), y[n_init_tr:].values.reshape(1,-1))
    accuracy = np.cumsum(pred_match)/np.arange(1, pred_match.shape[1]+1)
    # returns average accuracy, predicted y and true y
    return accuracy, np.array(y_pred).reshape(1,-1), y[n_init_tr:].values.reshape(1,-1)