import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

data = pd.read_csv('creditcard.csv')

data = data.sample(frac = 0.5 , random_state = 1)

Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outliners_frac = len(Fraud)/float(len(Valid))

columns = data.columns.tolist()

columns = [c for c in columns if c not in ["Class"]]
target = 'Class'

X = data[columns]
Y = data[target]

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest

state = 1

attr = IsolationForest(max_samples=len(X),
                       contamination = outliners_frac,
                       random_state = state)

attr.fit(X)
scores_pred = attr.decision_function(X)
y_pred = attr.predict(X)

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

error = sum(y_pred != Y)
print(accuracy_score(Y,y_pred))
print(classification_report(Y,y_pred))
