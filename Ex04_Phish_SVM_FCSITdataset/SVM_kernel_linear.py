import arff
import numpy as np
import time
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

start_time = time.perf_counter()

print("Importing dataset: FCSIT 2018 Phishing Examples")

dataset = arff.load(open('dataset_FCSIT_2018.arff', 'rt'))
data = np.array(dataset['data'])

print("Number of data points: ", data.shape[0])
print("Number of features: ", data.shape[1] - 1)

X = data[:, 0:48]
y = data[:, 48:49]

# split the data into test and train by maintaining same distribution of output variable y
print("Splitting dataset to TRAIN, CV, and TEST")
ts = 0.2  # test size
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=ts)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts)
unique_train, counts_train = np.unique(y_train, return_counts=True)
unique_test, counts_test = np.unique(y_test, return_counts=True)
print("Test dataset size: ", ts, ". Number of data points in:"
      "\n   TRAIN: ", X_train.shape[0], ", by classes: ", dict(zip(unique_train, counts_train)),
      "\n    TEST: ", X_test.shape[0], ", by classes: ", dict(zip(unique_test, counts_test)))

print("Training the model with Linear kernel")

svm = SVC(kernel='linear', gamma='auto', C=1)
svm.fit(X_train, np.ravel(y_train, order='C'))

print("Model trained")
print("Testing the model")

y_pred = svm.predict(X_test)

print("RESULTS:")
print("Confusion Matrix: \n |TP  FP| \n |FN  TN|\n\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report: \n", classification_report(y_test, y_pred))

print('Time took:', time.perf_counter() - start_time, "seconds")
