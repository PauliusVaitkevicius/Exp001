import arff
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


start_time = time.perf_counter()

print("Importing dataset: FCSIT 2018 Phishing Examples")

dataset = arff.load(open('dataset_FCSIT_2018.arff', 'rt'))
data = np.array(dataset['data'])

print("Number of data points: ", data.shape[0])
print("Number of features: ", data.shape[1] - 1)

X = data[:, 0:48]
y = data[:, 48:49]

print("Training the model with Gaussian kernel")
svm = SVC(kernel='rbf', C=1)
scores = cross_val_score(svm, X, np.ravel(y, order='C'), cv=10, scoring='accuracy', n_jobs=-1)
print(scores)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print('Time took:', time.perf_counter() - start_time, "seconds")
