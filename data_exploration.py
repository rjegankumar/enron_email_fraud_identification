#!/usr/bin/python

import pickle
import pprint
import matplotlib.pyplot as plt
import numpy as np
from feature_format import featureFormat, targetFeatureSplit

enron_data = pickle.load(open("final_project_dataset.pkl", "r"))

# length of the dictionary i.e. no. of training examples
print "\n# of training examples:", len(enron_data), "\n"

# no. of positive training examples or POIs
poi_count = 0
for name in enron_data:
    if enron_data[name]["poi"] == True:
        poi_count +=1
print "# of POIs in the training data:", poi_count, "\n"

# features in the training data
print "Features:\n"
features = enron_data[next(iter(enron_data))].keys()
features.remove("poi")
pprint.pprint(features)

'''
Based on intuition, some interesting features to work with:
    salary
    total_payments
    exercised_stock_options
    bonus
    restricted_stock
    shared_receipt_with_poi
    total_stock_value
    expenses
    loan_advances
    other
    from_this_person_to_poi
    long_term_incentive
    from_poi_to_this_person

After watching the documentary, features that could be important in identifying
POIs:
    exercised_stock_options
    restricted_stock
    bonus
    shared_receipt_with_poi
    from_this_person_to_poi
    from_poi_to_this_person
'''

# features with values as 'Not a Number'
print "\n# of NaN values for POI and non-POI in all features:\n"
for feature in features:
    nan_poi_count = 0
    nan_non_poi_count = 0
    for name in enron_data:
        if enron_data[name][feature] == "NaN":
            if enron_data[name]["poi"]:
                nan_poi_count += 1
            else:
                nan_non_poi_count += 1
    print feature, ":\n", "POI -", nan_poi_count, "\nnon-POI -", \
    nan_non_poi_count, "\n"
    
'''
After reviewing features with values as NaN, interesting features remain:
    total_payments
    total_stock_value
    shared_receipt_with_poi
    from_this_person_to_poi
    from_poi_to_this_person
'''

def scatter_plot(data, feature1, feature2):
    feature1_arr = np.array(create_list(data, feature1))
    feature2_arr = np.array(create_list(enron_data, feature2))
    poi_arr = np.array(create_list(enron_data, 'poi'))
    plot1 = plt.scatter(feature1_arr[poi_arr == False], 
                        feature2_arr[poi_arr == False], 
                        s = 50, c = "g", alpha = 0.5)
    plot2 = plt.scatter(feature1_arr[poi_arr == True], 
                        feature2_arr[poi_arr == True], 
                        s = 50, c = "r", alpha = 0.5)
    plt.legend((plot1, plot2), ("non-POI","POI"), loc="upper left")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.show()
    
# defining a function to create a list of values of certain features
def create_list(data, elm):
    temp_list = []
    for item in data:
        temp_list.append(data[item][elm])
    return temp_list

# plotting salary vs total_payments
scatter_plot(enron_data, "salary", "total_payments")

def outlier_detection(data, feature, threshold):
    for elm in data:
        if data[elm][feature] > threshold and data[elm][feature] != "NaN":
            print "Key of the", feature, "outlier:", elm

# identifying the outliers found in the previous scatter plot
outlier_detection(enron_data, 'total_payments', 100000000)
        
# removing 'TOTAL' from the data, need to remove from the poi_data code as well
del enron_data['TOTAL']
print "\n# of training examples after deleting 'TOTAL':", len(enron_data), "\n"

# re-plotting salary vs total_payments after removing the outlier
scatter_plot(enron_data, "salary", "total_payments")

# plotting more scatter plots to identify outliers
scatter_plot(enron_data, "bonus", "total_payments")
scatter_plot(enron_data, "expenses", "total_payments")
scatter_plot(enron_data, "loan_advances", "total_payments")
scatter_plot(enron_data, "other", "total_payments")
scatter_plot(enron_data, "long_term_incentive", "total_payments")
scatter_plot(enron_data, "exercised_stock_options", "total_stock_value")
scatter_plot(enron_data, "restricted_stock", "total_stock_value")

# identifying the outliers found in the previous scatter plot
outlier_detection(enron_data, 'total_stock_value', 40000000)
        
scatter_plot(enron_data, "shared_receipt_with_poi", "from_this_person_to_poi")
scatter_plot(enron_data, "from_poi_to_this_person", "from_this_person_to_poi")

# identifying the outliers found in the previous scatter plot
outlier_detection(enron_data, "from_this_person_to_poi", 500)
outlier_detection(enron_data, "from_poi_to_this_person", 500)