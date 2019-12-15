import time
import warnings

import arff
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB

warnings.filterwarnings("ignore")
start_time = time.perf_counter()

print("Importing dataset: FCSIT 2018 Phishing Examples")

dataset = arff.load(open('dataset_FCSIT_2018.arff', 'rt'))
data = np.array(dataset['data']).astype(np.float)

print("Number of data points: ", data.shape[0])
print("Number of features: ", data.shape[1] - 1)

X = data[:, 0:48]
y = data[:, 48:49]
cv = 30

model_label = "Naive-Bayes (Bernoulli)"
print("Training the " + model_label)

bnb = BernoulliNB()
scores = cross_val_score(bnb, X, np.ravel(y, order='C'), cv=cv, scoring='accuracy', n_jobs=-1)

# saving results to files
np.savetxt('../results/NB_CV30_normal.txt', scores)
scores.tofile('../results/NB_CV30_normal.dat')

print(scores)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Variance: ", np.var(scores))
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