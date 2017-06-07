from feature_format import featureFormat, targetFeatureSplit
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# loading the enron data dictionary
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# removing 'TOTAL' outlier
del data_dict['TOTAL']

# Adding new feature = ratio of from this person to POI and from messages
for name in data_dict:
    if data_dict[name]["from_this_person_to_poi"] != "NaN" and\
    data_dict[name]["from_messages"] != "NaN" and\
    data_dict[name]["from_messages"] != 0.0:
        data_dict[name]["toPOI_fromMsgs"] = \
        data_dict[name]["from_this_person_to_poi"] * 1.0\
        /data_dict[name]["from_messages"]
    else:
        data_dict[name]["toPOI_fromMsgs"] = 0.0

# function to train and predict POIs based on the features provided as input
def rfclassifier(feat_list):    
    # creating list of labels and list of numpy arrays containing the features
    data = featureFormat(data_dict, feat_list, sort_keys = True)
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
        
        clf = RandomForestClassifier(n_estimators=10, random_state=10)
        clf.fit(features_train, labels_train)
        pred = clf.predict(features_test)
        
        accuracy.append(clf.score(features_test, labels_test))
        precision.append(precision_score(labels_test, pred))
        recall.append(recall_score(labels_test, pred))
        f1.append(f1_score(labels_test, pred))
    
    print "\n\nNB Classifier Result with features - "
    for f in feat_list[1:]:
        print f
    print "\naccuracy:", np.mean(accuracy)
    print "precision:", np.mean(precision)
    print "recall:", np.mean(recall)
    print "f1 score:", np.mean(f1)
    
# selecting only 2 features - total_stock_value and bonus for now
# total_stock_value - data available for all POIs and second best feature
# bonus - data available for 16 out of 18 POIs and third best feature
rfclassifier(['poi',
              'total_stock_value',
              'bonus'])

# changing features list to replace total_stock_value with 
# exercised_stock_option
rfclassifier(['poi',
              'exercised_stock_options',
              'bonus'])
'''
Provided one of the best results with the highest precision and f1 scores
'''
    
# Adding saalary to the features list
rfclassifier(['poi',
              'exercised_stock_options',
              'bonus',
              'salary'])

# Removing everything except exercised stock options
rfclassifier(['poi',
              'exercised_stock_options'])
    
# Replacing exercised stock options with bonus
rfclassifier(['poi',
              'bonus'])
'''
Provided one of the best results with the precision and recall above 0.3
'''
    
# Replacing exercised stock options with bonus
rfclassifier(['poi',
              'salary'])
    
# Adding new feature to one of the best perfomring feature set
rfclassifier(['poi',
              'exercised_stock_options',
              'bonus',
              'toPOI_fromMsgs'])
'''
Best result thus far
'''

# Adding deferred income to the above list
rfclassifier(['poi',
              'exercised_stock_options',
              'bonus',
              'toPOI_fromMsgs',
              'deferred_income'])
'''
Improvement in accuracy, but a reduction in precision and f1 score
'''

# replacing deferred_income with long_term_incentive
rfclassifier(['poi',
              'exercised_stock_options',
              'bonus',
              'toPOI_fromMsgs',
              'long_term_incentive'])

# replacing long_term_incentive with restricted_stock    
rfclassifier(['poi',
              'exercised_stock_options',
              'bonus',
              'toPOI_fromMsgs',
              'restricted_stock'])