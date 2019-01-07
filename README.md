# MACHINE-LEARNING-DETECTING-ENRON-FRAUD

### PROJECT SUMMARY
Enron is one of the most famous fraud cases in history. In this project, I used supervised machine learning models to classify persons of interest (i.e. culpable persons) of the Enron case. I used both financial data and emails to achieve my target.

### PROJECT STRUCTURE

`Data Features`:            
salary
deferral_payments
total_payments
loan_advances
bonus
restricted_stock_deferred
deferred_income
total_stock_value
expenses
exercised_stock_options
other
long_term_incentive
restricted_stock
director_fees

`Email features`:
to_messages
email_address
from_poi_to_this_person
from_messages
from_this_person_to_poi
shared_receipt_with_poi

`POI labels`:
poi

The main script is `poi_id.py` which needs to be run first in order to generate the models for `tester.py` to yield the right results.

### PROJECT RESULTS
I tested 4 models to make a final choice: SVC, GaussianNB, DecisionTreeClf, KNeighborsClf.
KNeighborsClf was the best performing model. Namely, KNeighborsClassifer had 621 true positives and 12851 true negatives. Relative to 15000 datapoints this leads to an accuracy of 89.8%.In the context of KNeighborsClassifier, recall is not its strongest asset. SVC was better at it, but we looked at the aggregate performance of the model to choose KNeighborsClassifier as the best option.
KNeighborsClassifier had 621 true positives and 149 false positives. This leads to a precision of 80.6%.
KNeighborsClassifier has great precision. I.e if there is POI in the dataset, then it is very likely to be a true POI, and not a mistake. The low recall of KNeighborsClassifier implies that the model is
not guessing very well when it’s on the edge of POI classification.


