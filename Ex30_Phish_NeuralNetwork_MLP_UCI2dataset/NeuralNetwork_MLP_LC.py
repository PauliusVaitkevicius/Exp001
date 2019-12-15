import time
import warnings

import arff
import numpy as np
from sklearn.model_selection import ShuffleSplit

from utilities.plot_learning_curve import plot_learning_curve
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore")
start_time = time.perf_counter()

print("Importing dataset: UCI-2 2016 Phishing Examples")
dataset = arff.load(open('dataset_uci_2016.arff', 'rt'))
data = np.array(dataset['data']).astype(np.float)
X = data[:, 0:9]
y = data[:, 9:10]
y = np.ravel(y, order='C')
print("Number of data points: ", data.shape[0])
print("Number of features: ", data.shape[1] - 1)

# ------------------------------------------
model_label = "Multi-layer Perceptron (MLP) with Backpropagation"
estimator = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(30, ), learning_rate='adaptive', max_iter=3000)
# ------------------------------------------

print("Training the model: " + model_label)
cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=0)
plt = plot_learning_curve(estimator, "Learning Curves " + model_label, X, y,
                          ylim=(0.7, 1.01), cv=cv, n_jobs=-1)
plt.show()

print('Time took:', time.perf_counter() - start_time, "seconds")