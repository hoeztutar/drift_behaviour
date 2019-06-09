---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 1.0.2
  kernelspec:
    display_name: Concept Drift
    language: python
    name: env_cd
---

```python
import sys
sys.path.append("../")

import pandas as pd
pd.set_option('display.max_column', 250)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-notebook')
from matplotlib import rcParams
rcParams['figure.figsize'] = (6, 4)
rcParams['figure.dpi'] = 150

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

import arff
from tqdm import tnrange, tqdm_notebook

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
```

```python
data = arff.load(open('../data/elecNormNew.arff'))
cnames = [i[0] for i in data['attributes']]
df = pd.DataFrame(data['data'], columns=cnames)
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])
X = df.iloc[:,1:-1]
y = df['class']
```

```python
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
    return accuracy, y_pred, y[n_init_tr:]
```

```python
nb = GaussianNB()
lr = LogisticRegression(solver='sag')
svm = SVC(kernel='rbf', C=100, gamma=1)
tree = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=10, n_jobs=5)
forest = RandomForestClassifier(n_estimators=50)
gbc = GradientBoostingClassifier(n_estimators=50)
xgb = XGBClassifier()
```

```python
lr_res = prequential(40000, lr, X, y, 2000)
```

```python
plt.plot(lr_res)
plt.xticks([])
plt.legend('Logistic Regression')
```

```python

```

```python
toplo = sliding_prequential2(42000, model, X, y, 2000)
plt.plot(toplo)
```
