
# coding: utf-8

# ## Enron POI Identifier - 1.0


import pickle
import operator
import sys
sys.path.append("./Tools3/")

from featureFormat import featureFormat
from targetFeatureSplit import targetFeatureSplit
from getsetResultFiles import dump_classifier_and_data
from getsetResultFiles import load_classifier_and_data

from scoreFeatures import scoreTopFeatures
from addFeatures import poi_interaction_ratios
from validate import test_random_data_splits
from validate import test_classifier

from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


feature_list = [
    "poi",
    "deferral_payments",
    "deferred_income",
    "director_fees",
    "exercised_stock_options",
    "expenses",
    "from_messages",
    "from_poi_to_this_person",
    "from_this_person_to_poi",
    "long_term_incentive",
    "loan_advances",
    "other",
    "bonus",
    "restricted_stock",
    "restricted_stock_deferred",
    "salary",
    "shared_receipt_with_poi",
    "to_messages",
    "total_payments",
    "total_stock_value"]


# load data
data_dict = pickle.load((open("./data/final_project_dataset.pkl", "rb")),fix_imports=True)


# Remove outliers/fix records
data_dict.pop("TOTAL", 0 )


# Valid data per feature
non_nan = {}
for feature in feature_list:
    for k, v in data_dict.items():
        if v[feature] != 'NaN':
            if feature not in non_nan.keys():
                non_nan[feature] = 0
            non_nan[feature] += 1
            
valid_records_per_feature = sorted((non_nan).items(), key = operator.itemgetter(1), reverse = True)
print (valid_records_per_feature)


# Create new features
data_dict = poi_interaction_ratios(data_dict)
feature_list.append("to_poi_ratio")
feature_list.append("from_poi_ratio")
print("new features added: to_poi_ratio, from_poi_ratio")

# compute score of top 20 features
scoreTopFeatures(data_dict, feature_list, 20)

# select top 5 features, re-insert POI in front
feature_list = scoreTopFeatures(data_dict, feature_list, 5)
feature_list.insert(0, 'poi')
print("feature list: ", feature_list)

# scale all features
scaler = MinMaxScaler()

classifiers = [{'classifier': LogisticRegression(),
                'params': {  "clf__C": [0.05, 0.08, 0.1, 0.5, 1, 10, 10**2, 10**3, 10**5, 10**10, 10**15],
                    "clf__tol":[10**-1, 10**-2, 10**-4, 10**-5, 10**-6, 10**-10, 10**-15],
                    "clf__class_weight":['balanced']
                    }},
               {'classifier': DecisionTreeClassifier(),
                'params':
                    {
                        "clf__criterion": ["gini", "entropy"],
                        "clf__min_samples_split": [10,15,20,25]
                    }},
              {'classifier': KNeighborsClassifier(),
                'params': { "clf__n_neighbors":[1,2,3,4,5,6]
                            #"clf__weights": ["uniform", "distance"],
                            #"clf__algorithm": ["ball_tree", "kd_tree", "brute", "auto"]
                    }}]


my_dataset = data_dict
my_features_list = feature_list

### Extract features and labels from dataset for local testing
data = featureFormat(data_dict, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Find best classifier and best parameters
for c in classifiers:
    clf = Pipeline(steps=[("scaler", scaler), ("skb", SelectKBest(k='all')), ("clf", c['classifier'])])
    test_random_data_splits(clf, my_dataset, feature_list, features, labels, c['params'], 25)


# Select final classifier and hyper parameters
clf = Pipeline(steps=[("scaler", scaler),
                      ("skb", SelectKBest(k='all')),
                      ("clf", LogisticRegression(tol=0.1, C = 0.06, class_weight='balanced'))])

### Dump classifier, dataset, and features_list so anyone can
### check results.

dump_classifier_and_data(clf, my_dataset, feature_list)


### test classifier, print metrics and confusion matrix
tclf, tdataset, tfeature_list = load_classifier_and_data()
test_classifier(tclf, tdataset, tfeature_list, conf_matrix_filename="logisticRegressionCM", title="Logistic Regression Confusion Matrix")


