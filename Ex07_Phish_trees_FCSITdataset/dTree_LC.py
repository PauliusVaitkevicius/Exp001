import time

import arff
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from utilities.plot_learning_curve import plot_learning_curve

start_time = time.perf_counter()

print("Importing dataset: FCSIT 2018 Phishing Examples")

dataset = arff.load(open('dataset_FCSIT_2018.arff', 'rt'))
data = np.array(dataset['data'])

print("Number of data points: ", data.shape[0])
print("Number of features: ", data.shape[1] - 1)

X = data[:, 0:48]
y = data[:, 48:49]

print("Training the model with Decision Tree")
estimator = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=5, min_samples_leaf=2)

cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)

plt = plot_learning_curve(estimator, "Learning Curves (CART decision Tree)", X, np.ravel(y, order='C'),
                          ylim=(0.8, 1.01), cv=cv, n_jobs=-1)
plt.show()
print('Time took:', time.perf_counter() - start_time, "seconds")
