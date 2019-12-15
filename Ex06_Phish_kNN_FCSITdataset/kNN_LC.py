import time
import warnings

import arff
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from utilities.plot_learning_curve import plot_learning_curve

warnings.filterwarnings("ignore")
start_time = time.perf_counter()

print("Importing dataset: FCSIT 2018 Phishing Examples")

dataset = arff.load(open('dataset_FCSIT_2018.arff', 'rt'))
data = np.array(dataset['data'])
data = data.astype(np.float)

print("Number of data points: ", data.shape[0])
print("Number of features: ", data.shape[1] - 1)

X = data[:, 0:48]
y = data[:, 48:49]

# nn = round(math.sqrt(data.shape[0]), 0)
# if nn % 2 == 0:
#     nn += 1

nn = 5

model_label = "kNN (N=" + str(nn) + ")"
print("Training the " + model_label)

estimator = KNeighborsClassifier(n_neighbors=nn)

cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)

plt = plot_learning_curve(estimator, "Learning Curves (kNN (N=" + str(nn) + "))", X, np.ravel(y, order='C'),
                          ylim=(0.75, 1.01), cv=cv, n_jobs=-1)
plt.show()
print('Time took:', time.perf_counter() - start_time, "seconds")
