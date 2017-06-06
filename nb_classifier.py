from feature_format import featureFormat, targetFeatureSplit
import pickle
from sklearn.naive_bayes import GaussianNB

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
clf = GaussianNB()
clf.fit(features, labels)
print clf.score(features, labels)