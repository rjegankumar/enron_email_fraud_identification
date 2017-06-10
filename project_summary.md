# Identifying POIs in the Enron scandal

The objective of this project is to identify Persons of Interests in the well known Enron corporate scandal. Persons of Interests are those who have been indicted, pleaded guilty, confessed etc. Utilizing historical email and financial data of Enron's employees, I used supervised machine learning algorithms to identify Persons of Interests in the given set of employees.

The dataset used in this analysis is a dictionary with email and financial data of 146 employees. Out of these 146 employees, 18 are identified as POIs. The key of each element in the data dictionary is the name of an employee. The value of each element is another dictionary containing 20 possible features and 1 label. Missing feature values contain a "NaN" string, which is replaced with 0 before feeding into the machine learning model. There was one outlier in the data named "TOTAL", which was the total financials of all employees, this was removed from data dictionary prior to analysis.

## Feature Engineering and Selection

I created 4 new features:

- Total payments + total stock values
- Shared receipt with POI/ to messages
- From this person to POI/ from messages
- From POI to this person/ to messages
