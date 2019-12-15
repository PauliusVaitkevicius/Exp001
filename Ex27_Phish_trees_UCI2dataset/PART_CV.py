import time

import weka.core.jvm as jvm
from weka.attribute_selection import ASEvaluation, ASSearch, AttributeSelection
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
cls = Classifier(classname="weka.classifiers.rules.PART")
print(cls.options)
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1))
print("All attributes: %0.2f%%" % evl.percent_correct)
print(evl.confusion_matrix)

# aseval = ASEvaluation(classname="weka.attributeSelection.WrapperSubsetEval",
#                       options=["-F", "10", "-B", "weka.classifiers.trees.J48"])
# assearch = ASSearch(classname="weka.attributeSelection.BestFirst",
#                     options=["-D", "0", "-N", "5"])
# print("\n--> Attribute selection (cross-validation)\n")
# print(aseval.to_commandline())
# print(assearch.to_commandline())
# attsel = AttributeSelection()
# print("Step-1")
# attsel.evaluator(aseval)
# print("Step-2")
# attsel.search(assearch)
# print("Step-3")
# attsel.crossvalidation(True)
# print("Step-4")
# attsel.select_attributes(data)
# print("Step-5")
# print(attsel.results_string)

jvm.stop()
print('Time took:', time.perf_counter() - start_time, "seconds")
