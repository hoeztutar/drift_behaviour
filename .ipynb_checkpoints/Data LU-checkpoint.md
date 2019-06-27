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

# Data Luxembourg


## Importing Necessary Libraries and Reading Data

```python
from mat4py import loadmat
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.style.use('seaborn-notebook')
pd.set_option('display.max_column', 250)
from matplotlib import rcParams
rcParams['figure.figsize'] = (5, 3)
rcParams['figure.dpi'] = 150
```

```python
data = loadmat("data/dataLU.mat")
#print(data)
X = pd.DataFrame(data['dataLU']['X']).transpose()
y = pd.DataFrame(data['dataLU']['y'], columns=['label'])
t = pd.DataFrame(data['dataLU']['t'], columns=['date'])
print(X.shape, y.shape, t.shape)
```

```python
matlab_datenum = 1257.5833333333333
python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)
python_datetime
```

```python
t['date'].apply(lambda matlab_datenum: datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366))
```

```python
python_datetime
```

```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression().fit(X[:100], y.label[:100])
```

```python
logreg.score(X[:-100], y.label[:-100])
```

## Info on Dataset


**Name**:

Author/Complier:

Source:

Brief Explanation:

Shape:

No of Class:

Imbalance:
