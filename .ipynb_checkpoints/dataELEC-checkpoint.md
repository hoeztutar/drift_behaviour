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
```

### Read Data

```python
import arff
data = arff.load(open('data/elecNormNew.arff'))
cnames = [i[0] for i in data['attributes']]
df = pd.DataFrame(data['data'], columns=cnames)
df.head()
```

```python
df.describe()
```

```python
sns.pairplot(df, hue='class')
```

### Preprocess

```python
from sklearn.preprocessing import LabelEncoder
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

```python
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=42)
```

```python
param_grid = {'n_estimators': [5, 10, 100, 150, 200], 'min_samples_split': [5, 10, 20, 25, 50, 75, 100]}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid.fit(train_X, train_y)
grid.score(val_X, val_y)
```

## Naive Bayes


## Support Vector Machine


### Grid Search

```python
param_grid = {
    'kernel': ['rbf', 'poly', 'sigmoid'], 
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'gamma': ['auto', 0.001, 0.01, 0.1, 1]
}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(), param_grid, cv=3)
grid.fit(train_X, train_y)
print(grid.score(val_X, val_y))
```

### Model Deployment

```python
from sklearn.svm import SVC

SupVec = SVC()
SupVec.fit(train_X, train_y)
```

### Model Evaluation

```python
print("Training set score: {:.3f}".format(SupVec.score(train_X, train_y)))
print("Test set score: {:.3f}".format(SupVec.score(val_X, val_y)))
```

```python
from sklearn.model_selection import cross_val_score 


scores = cross_val_score(SupVec, X, y, cv=5)
print("Cross-validation scores: {}".format(scores))
```

## Logistic Regression


### Grid Search

```python
%%time
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty':['l1', 'l2']
    }

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(train_X, train_y)
grid.score(val_X, val_y)
```

### Model Deployment

```python
logreg = LogisticRegression
logreg.fit(train_X, train_y)
```

### Model Evaluation

```python
print("Training set score: {:.3f}".format(logreg.score(train_X, train_y)))
print("Test set score: {:.3f}".format(logreg.score(val_X, val_y)))
```

```python
from sklearn.model_selection import cross_val_score 


scores = cross_val_score(logreg, X, y, cv=5)
print("Cross-validation scores: {}".format(scores))
```
