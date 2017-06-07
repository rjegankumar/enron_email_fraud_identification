#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
nb_features_list = ['poi','exercised_stock_options'] # You will need to use more features
rf_features_list = ['poi',
                    'exercised_stock_options',
                    'bonus',
                    'toPOI_fromMsgs',
                    'sharedReceipt_toMsgs']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
del data_dict['TOTAL']

for name in data_dict:
    if data_dict[name]["from_this_person_to_poi"] != "NaN" and\
    data_dict[name]["from_messages"] != "NaN" and\
    data_dict[name]["from_messages"] != 0.0:
        data_dict[name]["toPOI_fromMsgs"] = \
        data_dict[name]["from_this_person_to_poi"] * 1.0\
        /data_dict[name]["from_messages"]
    else:
        data_dict[name]["toPOI_fromMsgs"] = 0.0
        
    if data_dict[name]["shared_receipt_with_poi"] != "NaN" and\
    data_dict[name]["to_messages"] != "NaN" and\
    data_dict[name]["to_messages"] != 0.0:
        data_dict[name]["sharedReceipt_toMsgs"] = \
        data_dict[name]["shared_receipt_with_poi"] * 1.0\
        /data_dict[name]["to_messages"]
    else:
        data_dict[name]["sharedReceipt_toMsgs"] = 0.0

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data_nb = featureFormat(my_dataset, nb_features_list, sort_keys = True)
labels_nb, features_nb = targetFeatureSplit(data_nb)

data_rf = featureFormat(my_dataset, rf_features_list, sort_keys = True)
labels_rf, features_rf = targetFeatureSplit(data_rf)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=15, 
                            criterion="gini",
                            min_samples_split=2,
                            min_samples_leaf=1,
                            random_state=10)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

accuracy_nb = []
precision_nb = []
recall_nb = []
f1_nb = []

kf = KFold(len(data_nb), n_folds=4, random_state=1)

for train_indices, test_indices in kf:
    features_train = [features_nb[i] for i in train_indices]
    features_test = [features_nb[j] for j in test_indices]
    labels_train = [labels_nb[i] for i in train_indices]
    labels_test = [labels_nb[j] for j in test_indices]
    
    nb.fit(features_train, labels_train)
    pred_nb = nb.predict(features_test)
    accuracy_nb.append(nb.score(features_test, labels_test))
    precision_nb.append(precision_score(labels_test, pred_nb))
    recall_nb.append(recall_score(labels_test, pred_nb))
    f1_nb.append(f1_score(labels_test, pred_nb))
    
print "\nNaive Bayes Evaluation Results -"
print "\naccuracy:", np.mean(accuracy_nb)
print "precision:", np.mean(precision_nb)
print "recall:", np.mean(recall_nb)
print "f1:", np.mean(f1_nb)

accuracy_rf = []
precision_rf = []
recall_rf = []
f1_rf = []

kf = KFold(len(data_rf), n_folds=4, random_state=1)

for train_indices, test_indices in kf:
    features_train = [features_rf[i] for i in train_indices]
    features_test = [features_rf[j] for j in test_indices]
    labels_train = [labels_rf[i] for i in train_indices]
    labels_test = [labels_rf[j] for j in test_indices]
    
    rf.fit(features_train, labels_train)
    pred_rf = rf.predict(features_test)
    accuracy_rf.append(rf.score(features_test, labels_test))
    precision_rf.append(precision_score(labels_test, pred_rf))
    recall_rf.append(recall_score(labels_test, pred_rf))
    f1_rf.append(f1_score(labels_test, pred_rf))

print "\n\nRandom Forest Evaluation Results -"
print "\naccuracy:", np.mean(accuracy_rf)
print "precision:", np.mean(precision_rf)
print "recall:", np.mean(recall_rf)
print "f1:", np.mean(f1_rf)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(rf, my_dataset, rf_features_list)