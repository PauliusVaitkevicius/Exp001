import time

import weka.core.jvm as jvm
from weka.classifiers import Classifier, Evaluation
from weka.core.classes import Random
from weka.core.converters import Loader

start_time = time.perf_counter()

print("Importing dataset: FCSIT 2018 Phishing Examples")

jvm.start()

fname = "dataset_FCSIT_2018.arff"

loader = Loader(classname="weka.core.converters.ArffLoader")
data = loader.load_file(fname)
data.class_is_last()

print("Training the model")
cls = Classifier(classname="weka.classifiers.trees.J48", options=["-C", "0.2"])
print(cls.options)
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 30, Random(1))

print("All attributes: %0.42f%%" % evl.percent_correct)
# print(evl.class_details())
# print(evl.confusion_matrix)
# print(evl.summary())

jvm.stop()
print('Time took:', time.perf_counter() - start_time, "seconds")
# http://weka.8497.n7.nabble.com/accuracy-value-for-each-cross-validation-td3487.html