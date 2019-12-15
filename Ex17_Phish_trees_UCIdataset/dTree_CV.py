import arff
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

start_time = time.perf_counter()

print("Importing dataset: UCI 2015 Phishing Examples")
dataset = arff.load(open('dataset_uci_2015.arff', 'rt'))
data = np.array(dataset['data'])
X = data[:, 0:30]
y = data[:, 30:31]
y = np.ravel(y, order='C')
print("Number of data points: ", data.shape[0])
print("Number of features: ", data.shape[1] - 1)

cv = 30

model_label = "CART decision tree"
print("Training the " + model_label)

dt = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=9, min_samples_leaf=2)

scores = cross_val_score(dt, X, y, cv=cv, scoring='accuracy')

# saving results to files
np.savetxt('../results/CART_CV30_normal_UCIdataset.txt', scores)

print("No of scores: ", scores.shape[0])
print(scores)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print('Time took:', time.perf_counter() - start_time, "seconds")

import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import qqplot
import math
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson

bins: int = round(1 + 3.322 * math.log10(scores.shape[0]), 0).__int__()
#bins = 10

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