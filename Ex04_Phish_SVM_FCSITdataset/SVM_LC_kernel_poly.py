import time

import arff
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from utilities.plot_learning_curve import plot_learning_curve


start_time = time.perf_counter()

print("Importing dataset: FCSIT 2018 Phishing Examples")

dataset = arff.load(open('dataset_FCSIT_2018.arff', 'rt'))
data = np.array(dataset['data'])

print("Number of data points: ", data.shape[0])
print("Number of features: ", data.shape[1] - 1)

X = data[:, 0:48]
y = data[:, 48:49]

print("Training the model with Polynomial kernel")
d = 1
estimator = SVC(kernel='poly', degree=d, C=1, gamma='auto')

cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)

title = "Learning Curves (SVM - Polynomial kernel | d = " + str(d) + ")"

plt = plot_learning_curve(estimator, title, X, np.ravel(y, order='C'),
                          ylim=(0.9, 0.95), cv=cv, n_jobs=-1)
plt.show()
print('Time took:', time.perf_counter() - start_time, "seconds")