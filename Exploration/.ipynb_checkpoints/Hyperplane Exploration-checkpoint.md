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
import pandas as pd
pd.set_option('display.max_column', 250)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-notebook')
from matplotlib import rcParams
rcParams['figure.figsize'] = (10, 5)
rcParams['figure.dpi'] = 150
```

```python
import arff
data = arff.load(open('../data/Hyperplane.arff'))
cnames = [i[0] for i in data['attributes']]
df = pd.DataFrame(data['data'], columns=cnames)
df.head()
```

```python
df.info()
```

```python
df.describe()
```

```python
df.groupby('class')['class'].count()
```

```python
f, ax = plt.subplots(5,2)

for t in range(2):
    for i in range(5):
        sns.distplot(df.iloc[:,i], ax=ax[i,t])
```

```python
#todo. this does not work. Make it classwise
class_ = 'groupB'
f, axes = plt.subplots(4, 2)

for i in range(2):
    sns.distplot(df[df['class']== class_].loc[:12500,i], ax=axes[0,i])
    sns.distplot(df[df['class']== class_].loc[12500:25000,i], ax=axes[1,i])
    sns.distplot(df[df['class']== class_].loc[25000:37500,i], ax=axes[2,i])
    sns.distplot(df[df['class']== class_].loc[37500:,i], ax=axes[3,i])
```

```python

```
