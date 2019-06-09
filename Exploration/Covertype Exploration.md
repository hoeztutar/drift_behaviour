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
rcParams['figure.figsize'] = (5, 3)
rcParams['figure.dpi'] = 150
```

```python
import arff
data = arff.load(open('../data/covtypeNorm.arff'))
cnames = [i[0] for i in data['attributes']]
df = pd.DataFrame(data['data'], columns=cnames)
df.head()
```

```python
df.describe()
```

```python
df.groupby('class')['class'].count()
```

```python
f, axes = plt.subplots(2, 2)

for i in range(2):
    sns.distplot(df.iloc[:25000,i], ax=axes[0,i])
    sns.distplot(df.iloc[25000:,i], ax=axes[1,i])
```

```python
len(df.columns)
```

```python
df.info()
```

```python
for i in df:
    print(i, df[i].nunique())
```

```python
plt.plot(df.index.sort_values(), df['class'], '.')
```
