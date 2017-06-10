import pickle
import matplotlib.pyplot as plt
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_classif
import pprint
import operator

# loading the enron data dictionary
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# removing 'TOTAL' outlier
del data_dict['TOTAL']

'''
Possible new features:
    total_payments + total_stock_value
    shared_receipt_with_poi/ to_messages
    from_this_person_to_poi/ from_messages
    from_poi_to_this_person/ to_messages
'''

# defining a function to return the labels and features as a numpy array
def labels_features(feature1, feature2):
    features_list = ['poi', feature1, feature2]
    data = featureFormat(data_dict, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    return np.array(labels), np.array(features)

# creating labels and features for the new features mentioned above
labels1, features1 = labels_features('total_payments', 'total_stock_value')
labels2, features2 = labels_features('shared_receipt_with_poi', 'to_messages')
labels3, features3 = labels_features('from_this_person_to_poi', 'from_messages')
labels4, features4 = labels_features('from_poi_to_this_person', 'to_messages')

# creating new features
new_features1 = features1[:,0] + features1[:,1]
new_features2 = features2[:,0] / features2[:,1]
new_features3 = features3[:,0] / features3[:,1]
new_features4 = features4[:,0] / features4[:,1]

# defining a function to create scatter plots
def scatter_plot(labels, features, feature_name):
    plt.scatter(labels,
                features,
                s = 50, c = "b", alpha = 0.5)
    plt.xlabel('poi')
    plt.ylabel(feature_name)
    plt.xticks(np.arange(0,1.5,1))
    plt.show()

# plotting new features vs poi
scatter_plot(labels1, new_features1, "total payments and stock value")
scatter_plot(labels2, new_features2, "shared poi receipt/ to messages")
scatter_plot(labels3, new_features3, "to poi/ from messages")
scatter_plot(labels4, new_features4, "from poi/ to messages")

# creating a list of all labels and features
all_features_list = ['poi',
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

# creating list of labels and list of numpy arrays containing the features
enron_data = featureFormat(data_dict, all_features_list, sort_keys = True)
enron_labels, enron_features = targetFeatureSplit(enron_data)

# adding new features from above to the list of numpy arrays
for i in np.arange(len(enron_features)):
    ttl_pay_stock = enron_features[i][3] + enron_features[i][9]
    enron_features[i] = np.append(enron_features[i], ttl_pay_stock)

    if enron_features[i][1] != 0:
        sharedReceipt_toMsgs = enron_features[i][7] * 1.0 / enron_features[i][1]
        enron_features[i] = np.append(enron_features[i], sharedReceipt_toMsgs)
    else:
        enron_features[i] = np.append(enron_features[i], 0.0)

    if enron_features[i][12] != 0:
        toPOI_fromMsgs = enron_features[i][14] * 1.0 / enron_features[i][12]
        enron_features[i] = np.append(enron_features[i], toPOI_fromMsgs)
    else:
        enron_features[i] = np.append(enron_features[i], 0.0)

    if enron_features[i][1] != 0:
        fromPOI_toMsgs = enron_features[i][18] * 1.0 / enron_features[i][1]
        enron_features[i] = np.append(enron_features[i], fromPOI_toMsgs)
    else:
        enron_features[i] = np.append(enron_features[i], 0.0)

# addining new feature names to the all features list
all_features_list.extend(["total_payments_stock",
                          "shared_receipt_poi_to_msgs",
                          "from_this_person_to_poi_from_msgs",
                          "from_poi_to_this_person_to_msgs"])

# listing out the best features in ascending order using SelectKBest method
best_features = {}
selector = SelectKBest(f_classif, k=23)
selector.fit(enron_features, enron_labels)
for i, s in enumerate(selector.scores_):
    best_features[all_features_list[(i+1)]] = s
pprint.pprint(sorted(best_features.items(), key=operator.itemgetter(1), reverse=True))
