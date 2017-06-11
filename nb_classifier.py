from feature_format import featureFormat, targetFeatureSplit
import pickle
from sklearn.naive_bayes import GaussianNB
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
              'from_poi_to_this_person']

# Selecting the best features using GridSearchCV
data = featureFormat(data_dict, feat_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

best_params = []
best_score = []
best_features = []

sss = StratifiedShuffleSplit(labels, 1000, random_state = 42)

for train_indices, test_indices in sss:
    features_train = [features[i] for i in train_indices]
    features_test = [features[j] for j in test_indices]
    labels_train = [labels[i] for i in train_indices]
    labels_test = [labels[j] for j in test_indices]

    pipe = Pipeline([('KBest', SelectKBest()),
                    ('clf', GaussianNB())])
    K = [1,2,3,4,5,6]
    param_grid = [{'KBest__k': K}]

    gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='f1')
    gs.fit(features_train, labels_train)

    best_params.append(gs.best_params_['KBest__k'])
    best_score.append(gs.best_score_)

    kb = SelectKBest(k=gs.best_params_['KBest__k'])
    kb.fit(features_train, labels_train)
    best_features.append(list(kb.get_support(indices=True)+1))

print "Best f1 score:", max(best_score)
print "No. of features used for the best f1 score:", best_params[best_score.index(max(best_score))]
print "Names of features used:\n", [feat_list[i] for i in best_features[best_score.index(max(best_score))]]

final_feat_list = ['poi', 'exercised_stock_options', 'bonus', 'total_stock_value']

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

    clf = GaussianNB()
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
