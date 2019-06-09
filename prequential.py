from time import clock

import numpy as np


def prequential(X, y, clf, n_train=1):
    """Prequential Evaluation: instances are first used to test, and then to train
    :return the label predictions for each test instance, and the associated running time
    """
    #Prequential evaliuation is used in SAMOA. More infot can be gathered in the link below
    #https://samoa.incubator.apache.org/documentation/Prequential-Evaluation-Task.html
    row_num = y.shape[0]
    # Split an init batch
    X_init = X[0:n_train]
    y_init = y[0:n_train]

    # Used for training and evaluation
    X_train = X[n_train:]
    y_train = y[n_train:]

    y_pre = np.zeros(row_num - n_train)
    time = np.zeros(row_num - n_train)

    #initiate model with the data defined by the number of n_train.
    clf.fit(X_init, y_init)

    #first predict the instannce and store in in an array then partial_fit it to clf.
    for i in range(0, row_num - n_train):
        start_time = clock()
        y_pre[i] = clf.predict(X_train.iloc[i, :].values.reshape(1, -1))
        clf.fit(X_train.iloc[i, :].values.reshape(1, -1), y_train[i].reshape(1, -1).ravel())
        time[i] = clock() - start_time

    return y_pre, time
