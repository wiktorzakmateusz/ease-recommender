# Quick start

To check the functionalities of the package, write the following code:

```python
from ease_recommender import EASE
import numpy as np
import scipy.sparse as sp

# URM of 5 users and 10 movies
URM = np.array([
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    [1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]
])

# convert to CSR sparse matrix
X_csr = sp.csr_matrix(URM)

# initialize with your preferred L2 regularization weight and number of CPU threads
model = EASE(reg_weight=100.0, num_threads=10)

# fit on a CSR sparse matrix
model.fit(X_csr)

# get Top-K recommendations for a user
user_history = [2, 7]
recommendations = model.predict_new_user(user_history, k=3)
print(recommendations)
```

then, the output should be:

```none
[1 8 3]
```