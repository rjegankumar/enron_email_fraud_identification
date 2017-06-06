from feature_format import featureFormat, targetFeatureSplit
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# loading the enron data dictionary
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# removing 'TOTAL' outlier
del data_dict['TOTAL']

# selecting only 2 features - total_stock_value and bonus for now
# total_stock_value - data available for all POIs and second best feature
# bonus - data available for 16 out of 18 POIs and third best feature
features_list = ['poi',
                 'total_stock_value',
                 'bonus']

# creating list of labels and list of numpy arrays containing the features
data = featureFormat(data_dict, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Fitting and testing Gaussian Naive Bayes Classifier
accuracy = []
precision = []
recall = []
f1 = []

kf = KFold(len(data), n_folds=4, random_state=1)
for train_indices, test_indices in kf:
    features_train = [features[i] for i in train_indices]
    features_test = [features[j] for j in test_indices]
    labels_train = [labels[i] for i in train_indices]
    labels_test = [labels[j] for j in test_indices]
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    accuracy.append(clf.score(features_test, labels_test))
    precision.append(precision_score(labels_test, pred))
    recall.append(recall_score(labels_test, pred))
    f1.append(f1_score(labels_test, pred))
    
print "accuracy:", np.mean(accuracy)
print "precision:", np.mean(precision)
print "recall:", np.mean(recall)
print "f1 score:", np.mean(f1)