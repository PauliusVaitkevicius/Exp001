import time
import warnings

import arff
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import AdaBoostClassifier

from utilities.plot_learning_curve import plot_learning_curve

warnings.filterwarnings("ignore")
start_time = time.perf_counter()

print("Importing dataset: UCI 2015 Phishing Examples")
dataset = arff.load(open('dataset_uci_2015.arff', 'rt'))
data = np.array(dataset['data'])
X = data[:, 0:30]
y = data[:, 30:31]
y = np.ravel(y, order='C')
print("Number of data points: ", data.shape[0])
print("Number of features: ", data.shape[1] - 1)

# ------------------------------------------
model_label = "AdaBoost"
estimator = AdaBoostClassifier(n_estimators=200, learning_rate=1)
# ------------------------------------------

print("Training the model: " + model_label)
cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
plt = plot_learning_curve(estimator, "Learning Curves " + model_label, X, y,
                          ylim=(0.9, 0.95), cv=cv, n_jobs=-1)
plt.show()

print('Time took:', time.perf_counter() - start_time, "seconds")