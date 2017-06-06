import pickle
import pprint
import matplotlib.pyplot as plt
import numpy as np
from feature_format import featureFormat, targetFeatureSplit

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