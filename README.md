# AutoGMM

Automatic Gaussian Mixture Modeling in Python.

```python
from autogmm import AutoGMM
labels = AutoGMM(min_components=1, max_components=10, random_state=0).fit_predict(X)

