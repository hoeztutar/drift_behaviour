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

### Import Libraries

```python
import pandas as pd
pd.set_option('display.max_column', 250)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-notebook')
from matplotlib import rcParams
rcParams['figure.figsize'] = (5, 3)
rcParams['figure.dpi'] = 150
%matplotlib inline

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import arff

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
```

### Read Data

```python
data = arff.load(open('../data/elecNormNew.arff'))
cnames = [i[0] for i in data['attributes']]
df = pd.DataFrame(data['data'], columns=cnames)
df.head()
```

```python
df.describe()
```

### Preprocess

```python
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])
```

```python
X = df.iloc[:,1:-1]
y = df['class']
```

```python
display(X.head())
display(y.head())
```

### Splitting data


Data is thought as batches and when every new batch arrives a new model is trained with the same parameters.

```python
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42, test_size=0.1, shuffle=False)
```

## Naive Bayes


### Model Deployment & Evaluation

```python
NB = GaussianNB()
NB.fit(train_X, train_y)

print("Training set score: {:.3f}".format(NB.score(train_X, train_y)))
print("Test set score: {:.3f}".format(NB.score(val_X, val_y)))

scores = cross_val_score(NB, X, y, cv=5)
print("Cross-validation scores: {}".format(scores))
```

```python

```

## Support Vector Machine


### Grid Search

```python
param_grid = {
    'kernel': ['rbf', 'poly', 'sigmoid'], 
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': ['auto', 0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(SVC(), param_grid, cv=5, verbose=10)
grid.fit(train_X, train_y)
print(grid.score(val_X, val_y))
```

0.7923728813559322
[CV]  kernel=rbf, C=100, gamma=1, score=0.7924084154774165, total=  21.9s


### Model Deployment & Evaluation

```python
SupVec = SVC()
SupVec.fit(train_X, train_y)

print("Training set score: {:.3f}".format(SupVec.score(train_X, train_y)))
print("Test set score: {:.3f}".format(SupVec.score(val_X, val_y)))

scores = cross_val_score(SupVec, X, y, cv=5)
print("Cross-validation scores: {}".format(scores))
```

## Logistic Regression


### Grid Search

```python
param_grid = {
    'C': [i for i in range(10, 51, 10)],
    }

grid = GridSearchCV(LogisticRegression(solver='sag', max_iter=3000), param_grid, cv=5)
grid.fit(train_X, train_y)
print("Score:", grid.score(val_X, val_y))
print("The best parameters:",grid.best_params_)
```

### Model Deployment & Evaluation

```python
logreg = LogisticRegression()
logreg.fit(train_X, train_y)

print("Training set score: {:.3f}".format(logreg.score(train_X, train_y)))
print("Test set score: {:.3f}".format(logreg.score(val_X, val_y)))

scores = cross_val_score(logreg, X, y, cv=5)
print("Cross-validation scores: {}".format(scores))
```

```python
param_grid = {
    'n_estimators': [i for i in range(10, 75, 5)]
    }

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid.fit(train_X, train_y)
print("Score:", grid.score(val_X, val_y))
print("The best parameters:",grid.best_params_)
```