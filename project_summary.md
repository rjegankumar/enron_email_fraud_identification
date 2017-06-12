# Identifying POIs in the Enron scandal

The objective of this project is to identify Persons of Interests in the well-known Enron corporate scandal. Persons of Interests are those who have been indicted, pleaded guilty, confessed etc. Utilizing historical email and financial data of Enron's employees, supervised machine learning algorithms were used to identify Persons of Interests in the given set of employees.

The dataset used in this analysis is a dictionary with email and financial data of 146 employees. Out of these 146 employees, 18 are identified as POIs. The key of each element in the data dictionary is the name of an employee. The value of each element is another dictionary containing 20 features and 1 label. Missing feature values contain a "NaN" string, which was replaced with 0 before feeding into the machine learning model. There was one outlier in the data named "TOTAL", which was the total financials of all employees, this was removed from data dictionary prior to analysis.

## Feature Engineering and Selection

1 new feature was created:

- Total payments + total stock values - based on the research which included the documentary, it was clear that the POIs received large financial compensations in terms of salaries, bonuses and stocks. So, the intuition was that this new feature could help identify POIs easily.

Then, SelectKBest univariate feature selection was used to list features in ascending order of importance. Features with SelectKBest scores:

```
[('exercised_stock_options', 25.097541528735491),
 ('total_stock_value', 24.467654047526398),
 ('bonus', 21.060001707536571),
 ('salary', 18.575703268041785),
 ('total_payments_stock', 17.187006077009126),
 ('deferred_income', 11.595547659730601),
 ('long_term_incentive', 10.072454529369441),
 ('restricted_stock', 9.3467007910514877),
 ('total_payments', 8.8667215371077717),
 ('shared_receipt_with_poi', 8.7464855321290802),
 ('loan_advances', 7.2427303965360181),
 ('expenses', 6.2342011405067401),
 ('from_poi_to_this_person', 5.3449415231473374),
 ('other', 4.204970858301416),
 ('from_this_person_to_poi', 2.4265081272428781),
 ('director_fees', 2.1076559432760908),
 ('to_messages', 1.6988243485808501),
 ('deferral_payments', 0.2170589303395084),
 ('from_messages', 0.16416449823428736),
 ('restricted_stock_deferred', 0.06498431172371151)]
```

As you can see above, there is a steep decline in scores from total_payments_stock to deferred_income. So, GridSearchCV was used for each of the algorithms to select a combination of 5 or fewer features that produced the highest f1 score.

- For the Naive Bayes classifier, salary, exercised stock options, bonus, and total stock value features were used.
- For the Random Forest classifier, exercised stock options and total stock value features were used.

The features did not need scaling for either of the algorithms as it wouldn't have had an effect.

## Machine Learning Algorithm and Tuning

As mentioned before, two algorithms for identifying POIs were used - Gaussian Naive Bayes and Random Forest. I used the Gaussian Naive Bayes as the final model as it gave the highest F1 score. One contrasting difference that was apparent was that the Naive Bayes classifier was much faster than Random Forest.

Tuning in a broad sense is to identify close to optimal conditions or options for the machine learning model to perform the best with the given data. Testing out different combinations of model parameters and observing the outputs will help us decipher which settings work best together. Tuning was an important step in achieving a higher F1 score for the Random Forest classifier. Several parameters were systematically tested using GridSearchCV - criterion, n_estimators, min_samples_split and min_samples_leaf, and final parameters selected were those that gave the highest F1 score:

- criterion = entropy
- n_estimators = 5
- min_samples_split = 8
- min_samples_leaf = 2

## Validation and Evaluation

Validation of a machine learning model is to test the trained model on an unseen or un-trained dataset. Testing the model on the trained data itself would yield results that are unrealistic and are less generalized due to over-fitting. StratifiedShuffleSplit with 1000 folds was used for both the Gaussian Naive Bayes and the Random Forest classifier to address the class imbalance problem.

4 evaluation metrics were used for both algorithms - accuracy, precision, recall and F1. But, the F1 score was the decisive metric. Final metrics for Naive Bayes were:

```
Evaluation Results for GaussianNB -

accuracy: 0.846769230769
precision: 0.4138
recall: 0.323
f1: 0.344376190476
```

Final metrics for Random Forest were -

```
Evaluation Results for RandomForestClassifier -

accuracy: 0.848692307692
precision: 0.3497
recall: 0.2485
f1: 0.277885714286
```

From above, clearly the naive bayes model was able to identify all POIs and non-POIs relatively accurately (based on accuracy), but this could be misleading as there are fewer POIs than non-POIs. This is where precision, recall and F1 scores are more valuable. Precision is how accurately the model identifies POIs i.e. the fraction when the model identifies an employee as a POI then this employee is in fact a POI. Recall is what fraction of the POIs did the model identify correctly as POIs. F1 score is a combination of both precision and recall, it increases when both precision and recall increase. In the above models though the accuracy is not nearly that great, precision, recall and F1 scores are high, based on the F1 scores alone, the Gaussian Naive Bayes classifier came out on top.
