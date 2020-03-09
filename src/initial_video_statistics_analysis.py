# SENG474 Group Project - Initial Analysis - Video statistics
# Amanda Munrp (V00802056)
# March 8th 2020

# %% Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 


# %% Load data

main_tbl = pd.read_csv('dataset.csv')
main_tbl = main_tbl[["likes", "dislikes", "views", "comment_count", "category_id"]]

# %% Preprocess 

X = main_tbl.iloc[:, [0, 1, 2, 3]]
y = main_tbl.iloc[:, 4]
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# %% Initial models 
np.random.seed(912)

clf = DecisionTreeClassifier(criterion = "gini")
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Decision Tree Classifier Accuracy:", metrics.accuracy_score(y_test, y_pred))

clf = RandomForestClassifier(criterion = 'gini', max_features = 3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Random Forest Classifier Accuracy:", metrics.accuracy_score(y_test, y_pred))

clf = MLPClassifier(hidden_layer_sizes = (10), max_iter = 1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Neural Net Accuracy:", metrics.accuracy_score(y_test, y_pred))

clf = LogisticRegression(penalty = "l2", C = 1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Logistic Regression Accuracy: ", metrics.accuracy_score(y_test, y_pred))

clf = svm.LinearSVC(C = 100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Support Vector Machine Accuracy: ", metrics.accuracy_score(y_test, y_pred))

# %% Tuning C parameter for Logistic Regression
np.random.seed(912)

# Tuning C parameter
c_ranges = np.logspace(-20, 20, base = 10, num = 41)
train_scores = []
test_scores = []
for c in c_ranges:
    clf = LogisticRegression(penalty = "l2", C = c)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

# Graph C training/test scores over C parameter
fig, ax = plt.subplots()
ax.set_xlabel("C Value")
ax.set_ylabel("Score")
ax.set_title("Score vs C value for training and testing sets")
ax.plot(c_ranges, train_scores,
    marker = '.',
    label = "Train")
ax.plot(c_ranges, test_scores, 
    marker = '.', 
    label = "Test")
ax.set_xscale('log')
ax.legend()
plt.show()


# %% Tuning C parameter for SVM
np.random.seed(912)

c_ranges = np.logspace(-15, 10, base = 10, num = 26)
train_scores = []
test_scores = []
for c in c_ranges:
    # print("c: " + str(c))
    clf = svm.LinearSVC(C = c)
    clf.fit(X_train, y_train)
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test, y_test))

# Graph C training/test scores over C parameter
fig, ax = plt.subplots()
ax.set_xlabel("C Value")
ax.set_ylabel("Score")
ax.set_title("Score vs C value for training and testing sets")
ax.plot(c_ranges, train_scores,
    marker = '.',
    label = "Train")
ax.plot(c_ranges, test_scores, 
    marker = '.', 
    label = "Test")
ax.set_xscale('log')
ax.legend()
plt.show()


# %%
