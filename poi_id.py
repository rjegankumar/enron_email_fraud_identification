import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
nb_features_list = ['poi',
                    'salary',
                    'exercised_stock_options',
                    'bonus',
                    'total_stock_value']
rf_features_list = ['poi',
                    'exercised_stock_options',
                    'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict['TOTAL']

### Task 3: Create new feature(s)
for name in data_dict:
    if data_dict[name]["total_payments"] != "NaN" and\
    data_dict[name]["total_stock_value"] != "NaN":
        data_dict[name]["ttl_pay_stock"] = \
        data_dict[name]["total_payments"] + \
        data_dict[name]["total_stock_value"]
    else:
        data_dict[name]["ttl_pay_stock"] = 0.0

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=5,
                            criterion="entropy",
                            random_state=20,
                            min_samples_split=8,
                            min_samples_leaf=2)

# evaluation
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def clf_evaluation(clf_name, clf, feat_list):
    data = featureFormat(my_dataset, feat_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    accuracy = []
    precision = []
    recall = []
    f1 = []

    sss = StratifiedShuffleSplit(labels, 1000, random_state = 42)

    for train_indices, test_indices in sss:
        features_train = [features[i] for i in train_indices]
        features_test = [features[j] for j in test_indices]
        labels_train = [labels[i] for i in train_indices]
        labels_test = [labels[j] for j in test_indices]

        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, pred))
        precision.append(precision_score(labels_test, pred))
        recall.append(recall_score(labels_test, pred))
        f1.append(f1_score(labels_test, pred))

    print "\nEvaluation Results for", clf_name, "-"
    print "\naccuracy:", np.mean(accuracy)
    print "precision:", np.mean(precision)
    print "recall:", np.mean(recall)
    print "f1:", np.mean(f1)

clf_evaluation("GaussianNB", nb, nb_features_list)
clf_evaluation("RandomForestClassifier", rf, rf_features_list)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(nb, my_dataset, nb_features_list)
