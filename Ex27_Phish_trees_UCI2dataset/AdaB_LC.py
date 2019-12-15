import time
import warnings

import arff
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import AdaBoostClassifier

from utilities.plot_learning_curve import plot_learning_curve

warnings.filterwarnings("ignore")
start_time = time.perf_counter()

print("Importing dataset: UCI-2 2016 Phishing Examples")
dataset = arff.load(open('dataset_uci_2016.arff', 'rt'))
data = np.array(dataset['data'])
X = data[:, 0:9]
y = data[:, 9:10]
y = np.ravel(y, order='C')
print("Number of data points: ", data.shape[0])
print("Number of features: ", data.shape[1] - 1)

# ------------------------------------------
model_label = "AdaBoost"

n_estimators = 50

estimator = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=1, algorithm='SAMME')
# ------------------------------------------

print("Training the model: " + model_label)
cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
plt = plot_learning_curve(estimator, "Learning Curves " + model_label + ", E=" + str(n_estimators), X, y,
                          ylim=(0.75, 0.90), cv=cv, n_jobs=-1)
plt.show()

print('Time took:', time.perf_counter() - start_time, "seconds")