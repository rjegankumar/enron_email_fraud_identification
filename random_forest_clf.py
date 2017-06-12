from feature_format import featureFormat, targetFeatureSplit
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV
import numpy as np

# loading the enron data dictionary
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# removing 'TOTAL' outlier
del data_dict['TOTAL']

# creating new features
for name in data_dict:
    if data_dict[name]["total_payments"] != "NaN" and\
    data_dict[name]["total_stock_value"] != "NaN":
        data_dict[name]["ttl_pay_stock"] = \
        data_dict[name]["total_payments"] + \
        data_dict[name]["total_stock_value"]
    else:
        data_dict[name]["ttl_pay_stock"] = 0.0

# list containing all labels and features except email
feat_list = ['poi',
              'salary',
              'to_messages',
              'deferral_payments',
              'total_payments',
              'exercised_stock_options',
              'bonus',
              'restricted_stock',
              'shared_receipt_with_poi',
              'restricted_stock_deferred',
              'total_stock_value',
              'expenses',
              'loan_advances',
              'from_messages',
              'other',
              'from_this_person_to_poi',
              'director_fees',
              'deferred_income',
              'long_term_incentive',
              'from_poi_to_this_person',
              'ttl_pay_stock']

# Selecting the best features using GridSearchCV
data = featureFormat(data_dict, feat_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

pipe = Pipeline([('KBest', SelectKBest()),
                ('clf', RandomForestClassifier())])
param_grid = [{'KBest__k': [1,2,3,4,5],
                'clf__n_estimators': [5,10,15,20,25],
                'clf__criterion': ['entropy','gini'],
                'clf__random_state': [20],
                'clf__min_samples_split': [2,4,8,16,32],
                'clf__min_samples_leaf': [1,2,4,8,16]}]

gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='f1')
gs.fit(features, labels)

kb = SelectKBest(k=gs.best_params_['KBest__k'])
kb.fit(features, labels)
best_feat = list(kb.get_support(indices=True)+1)

print "Best f1 score:", gs.best_score_
print "No. of features used for the best f1 score:", gs.best_params_['KBest__k']
print "Names of features used:\n", [feat_list[i] for i in best_feat]
print "No. of estimators used:", gs.best_params_['clf__n_estimators']
print "Criterion used:", gs.best_params_['clf__criterion']
print "Min samples split used:", gs.best_params_['clf__min_samples_split']
print "Min samples leaf used:", gs.best_params_['clf__min_samples_leaf']

final_feat_list = ['poi',
                    'exercised_stock_options',
                    'bonus',
                    'total_stock_value']

# Computing evaluation metrics using the selected features
final_data = featureFormat(data_dict, final_feat_list, sort_keys = True)
final_labels, final_features = targetFeatureSplit(final_data)

final_sss = StratifiedShuffleSplit(final_labels, 1000, random_state = 42)

accuracy = []
precision = []
recall = []
f1 = []

for train_indices, test_indices in final_sss:
    features_train = [final_features[i] for i in train_indices]
    features_test = [final_features[j] for j in test_indices]
    labels_train = [final_labels[i] for i in train_indices]
    labels_test = [final_labels[j] for j in test_indices]

    clf = RandomForestClassifier(n_estimators=15,
                                    criterion="gini",
                                    random_state=20)
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    accuracy.append(accuracy_score(labels_test, pred))
    precision.append(precision_score(labels_test, pred))
    recall.append(recall_score(labels_test, pred))
    f1.append(f1_score(labels_test, pred))

print "Evaluation results of GaussianNB using best features:"
print "Mean Accuracy:", np.mean(accuracy)
print "Mean Precision:", np.mean(precision)
print "Mean Recall:", np.mean(recall)
print "Mean f1 score:", np.mean(f1)


# final_feat_list = ['poi', 'exercised_stock_options', 'bonus', 'total_stock_value']
#
# # Computing evaluation metrics using the selected features
# final_data = featureFormat(data_dict, final_feat_list, sort_keys = True)
# final_labels, final_features = targetFeatureSplit(final_data)
#
# final_sss = StratifiedShuffleSplit(final_labels, 1000, random_state = 42)
#
# accuracy = []
# precision = []
# recall = []
# f1 = []
#
# for train_indices, test_indices in final_sss:
#     features_train = [final_features[i] for i in train_indices]
#     features_test = [final_features[j] for j in test_indices]
#     labels_train = [final_labels[i] for i in train_indices]
#     labels_test = [final_labels[j] for j in test_indices]
#
#     clf = GaussianNB()
#     clf.fit(features_train, labels_train)
#     pred = clf.predict(features_test)
#
#     accuracy.append(accuracy_score(labels_test, pred))
#     precision.append(precision_score(labels_test, pred))
#     recall.append(recall_score(labels_test, pred))
#     f1.append(f1_score(labels_test, pred))
#
# print "Evaluation results of GaussianNB using best features:"
# print "Mean Accuracy:", np.mean(accuracy)
# print "Mean Precision:", np.mean(precision)
# print "Mean Recall:", np.mean(recall)
# print "Mean f1 score:", np.mean(f1)



# # function to train and predict POIs based on the features provided as input
# def rfclassifier(feat_list, c="gini", n_est=10, min_split=2, min_leaf=1):
#     # creating list of labels and list of numpy arrays containing the features
#     data = featureFormat(data_dict, feat_list, sort_keys = True)
#     labels, features = targetFeatureSplit(data)
#
#     # Fitting and testing Gaussian Naive Bayes Classifier
#     accuracy = []
#     precision = []
#     recall = []
#     f1 = []
#
#     sss = StratifiedShuffleSplit(labels, 1000, random_state = 1)
#
#     for train_indices, test_indices in sss:
#         features_train = [features[i] for i in train_indices]
#         features_test = [features[j] for j in test_indices]
#         labels_train = [labels[i] for i in train_indices]
#         labels_test = [labels[j] for j in test_indices]
#
#         clf = RandomForestClassifier(n_estimators=n_est,
#                                      criterion=c,
#                                      min_samples_split=min_split,
#                                      min_samples_leaf=min_leaf,
#                                      random_state=10)
#         clf.fit(features_train, labels_train)
#         pred = clf.predict(features_test)
#
#         accuracy.append(clf.score(features_test, labels_test))
#         precision.append(precision_score(labels_test, pred))
#         recall.append(recall_score(labels_test, pred))
#         f1.append(f1_score(labels_test, pred))
#
#     print "\n\nRF Classifier with FEATURES - "
#     for f in feat_list[1:]:
#         print f
#     print "\nPARAMETERS -"
#     print "n_estimators:", n_est
#     print "criterion:", c
#     print "min_samples_split:", min_split
#     print "min_samples_leaf:", min_leaf
#     print "\nEVALUATION RESULTS -"
#     print "accuracy:", np.mean(accuracy)
#     print "precision:", np.mean(precision)
#     print "recall:", np.mean(recall)
#     print "f1:", np.mean(f1)

## selecting only 2 features - total_stock_value and bonus for now
## total_stock_value - data available for all POIs and second best feature
## bonus - data available for 16 out of 18 POIs and third best feature
#rfclassifier(['poi',
#              'total_stock_value',
#              'bonus'])
#
## changing features list to replace total_stock_value with
## exercised_stock_option
#rfclassifier(['poi',
#              'exercised_stock_options',
#              'bonus'])
#'''
#Provided one of the best results with the highest f1 score
#'''
#
## Adding saalary to the features list
#rfclassifier(['poi',
#              'exercised_stock_options',
#              'bonus',
#              'salary'])
#
## Removing everything except exercised stock options
#rfclassifier(['poi',
#              'exercised_stock_options'])
#
## Replacing exercised stock options with bonus
#rfclassifier(['poi',
#              'bonus'])
#'''
#Provided one of the best results with the highest f1 score
#'''
#
## Replacing exercised stock options with bonus
#rfclassifier(['poi',
#              'salary'])
#
## Adding new feature to one of the best perfomring feature set
#rfclassifier(['poi',
#              'exercised_stock_options',
#              'bonus',
#              'toPOI_fromMsgs'])
#'''
#Best result thus far
#'''
#
## Adding deferred income to the above list
#rfclassifier(['poi',
#              'exercised_stock_options',
#              'bonus',
#              'toPOI_fromMsgs',
#              'deferred_income'])
#'''
#Improvement in accuracy, but a reduction in precision and f1 score
#'''
#
## replacing deferred_income with long_term_incentive
#rfclassifier(['poi',
#              'exercised_stock_options',
#              'bonus',
#              'toPOI_fromMsgs',
#              'long_term_incentive'])
#
## replacing long_term_incentive with restricted_stock
#rfclassifier(['poi',
#              'exercised_stock_options',
#              'bonus',
#              'toPOI_fromMsgs',
#              'restricted_stock'])
#
## replacing restricted_stock with new feature ratio of shared receipts with POI
## and to messages
#rfclassifier(['poi',
#              'exercised_stock_options',
#              'bonus',
#              'toPOI_fromMsgs',
#              'sharedReceipt_toMsgs'])
#'''
#Best of all, improvement to precision and f1 score compared to previous best
#'''
#
## Tuning the above model with different criterion
#rfclassifier(['poi',
#              'exercised_stock_options',
#              'bonus',
#              'toPOI_fromMsgs',
#              'sharedReceipt_toMsgs'], "entropy")
#
## Using criterion = entropy, and tuning n_estimators
#for i in [1,5,10,15,20,50]:
#    rfclassifier(['poi',
#                  'exercised_stock_options',
#                  'bonus',
#                  'toPOI_fromMsgs',
#                  'sharedReceipt_toMsgs'], "entropy", i)
#
## Using criterion = entropy, n_estimators = 10, tuning min_samples_split
#for i in [2,4,8,16]:
#    rfclassifier(['poi',
#                  'exercised_stock_options',
#                  'bonus',
#                  'toPOI_fromMsgs',
#                  'sharedReceipt_toMsgs'], "entropy", 10, i)
#
## Using criterion = entropy, n_estimators = 10, min_samples_split = 2
## tuning min_samples_leaf
#for i in [1,2,4,8]:
#    rfclassifier(['poi',
#                  'exercised_stock_options',
#                  'bonus',
#                  'toPOI_fromMsgs',
#                  'sharedReceipt_toMsgs'], "entropy", 10, 2, i)
#
# '''
# final params:
#     criterion = gini
#     n_estimators = 15
#     min_samples_split = 2
#     min_samples_leaf = 1
# '''
# print "\n\nFINAL MODEL:"
# rfclassifier(['poi',
#               'exercised_stock_options',
#               'bonus',
#               'toPOI_fromMsgs',
#               'sharedReceipt_toMsgs'], "gini", 15)
