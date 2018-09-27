import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat

data = loadmat('ex8data2.mat')

X = data['X']
Xval = data['Xval']
Yval = data['yval']

Fraud = sum(Yval == 1)
Valid = sum(Yval == 0)

outliners_frac = Fraud/float(Valid)

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest

state = 1

attr = IsolationForest(max_samples=len(X),
                       contamination = outliners_frac,
                       random_state = state)

attr.fit(X)             
scores_pred = attr.decision_function(Xval)
y_pred = attr.predict(Xval)

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1

error = sum(y_pred == 1)
print(accuracy_score(Yval,y_pred))
print(classification_report(Yval,y_pred))
