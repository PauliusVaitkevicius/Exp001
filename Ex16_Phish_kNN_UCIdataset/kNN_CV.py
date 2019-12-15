import time
import warnings

import arff
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

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

# nn = round(math.sqrt(data.shape[0]), 0)
# if nn % 2 == 0:
#     nn += 1

nn = 5
cv = 30

model_label = "kNN (N=" + str(nn) + ")"
print("Training the " + model_label)


knn = KNeighborsClassifier(n_neighbors=nn)
scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
print(scores)

# saving results to files
np.savetxt('../results/kNN_N5_normal_UCIdataset.txt', scores)


print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import math
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson

# bins: int = round(1 + 3.322 * math.log10(scores.shape[0]), 0).__int__()
bins = 10

print("Bins No (Sturge’s Rule): ", bins)
plt.hist(scores, bins=bins)
plt.ylabel('Probability')
plt.xlabel("Accuracy")
plt.title("Accuracy of " + model_label + " with CV=" + str(cv))
plt.show()
qqplot(scores, line='s')
plt.title("Accuracy of " + model_label + " with CV=" + str(cv))
plt.show()

alpha = 0.05

print("Shapiro-Wilk Test result:")
stat, p = shapiro(scores)
print('     Statistics=%.3f, p=%.3f, alpha=%.3f' % (stat, p, alpha))
if p > alpha:
    print('     Sample looks Gaussian (fail to reject H0)')
else:
    print('     Sample does not look Gaussian (reject H0)')

print("D’Agostino’s K^2 Test result:")
stat, p = normaltest(scores)
print('     Statistics=%.3f, p=%.3f, alpha=%.3f' % (stat, p, alpha))
if p > alpha:
    print('     Sample looks Gaussian (fail to reject H0)')
else:
    print('     Sample does not look Gaussian (reject H0)')

print("Anderson-Darling Test result:")
result = anderson(scores)
print('     Statistic: %.3f' % result.statistic)
p = 0
for i in range(len(result.critical_values)):
    sl, c = result.significance_level[i], result.critical_values[i]
    if result.statistic < result.critical_values[i]:
        print('     %.3f: %.3f, data looks normal (fail to reject H0)' % (sl, c))
    else:
        print('     %.3f: %.3f, data does not look normal (reject H0)' % (sl, c))

print('Time took:', time.perf_counter() - start_time, "seconds")

# ---------------------------------------------------------------------------
#
# k_range = list(range(1, 21, 2))
# k_scores = []
# for k in k_range:
#     print("Iteration: ", k)
#     knn = KNeighborsClassifier(n_neighbors=k, p='2', metric='euclidean', n_jobs=-1)
#     scores = cross_val_score(knn, X, y, cv=30, scoring='accuracy')
#     k_scores.append(scores.mean())
# print(k_scores)
#
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated Accuracy')
# plt.grid(True)
# plt.show()
