import numpy as np
import pandas as pd

from collections import defaultdict

from sklearn import preprocessing, utils
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_predict

data = pd.read_csv(
    'data/match_matrix.csv',
    delimiter=', ',
    engine='python'
).dropna(axis=0)

def invert(df):
    to_return = df.copy()
    to_return['WRank'] = df['LRank']
    to_return['LRank'] = df['WRank']
    to_return['WElapsed'] = df['LElapsed']
    to_return['LElapsed'] = df['WElapsed']
    to_return['WRClay'] = df['LRClay']
    to_return['WRGrass'] = df['LRGrass']
    to_return['WRHard'] = df['LRHard']
    to_return['LRClay'] = df['WRClay']
    to_return['LRGrass'] = df['WRGrass']
    to_return['LRHard'] = df['WRHard']
    to_return['WAvgSpeed'] = df['LAvgSpeed']
    to_return['LAvgSpeed'] = df['WAvgSpeed']
    to_return['Wsets'] = df['Lsets']
    to_return['Lsets'] = df['Wsets']
    to_return['WDigit'] = 0
    to_return['LDigit'] = 1
    return to_return

first = data.head(len(data)//2)
second = data.tail((len(data)-len(first)))
second = invert(second)
data = pd.concat([first, second])
data.drop(['Winner', 'Loser', 'Wsets', 'Lsets'], inplace=True, axis=1)

data = pd.concat([
    data,
    pd.get_dummies(data['Surface']),
    pd.get_dummies(data['Court']),
    pd.get_dummies(data['Round']),
    pd.get_dummies(data['Series'])
], axis=1)

data.drop(['Surface', 'Court', 'Round', 'Series'], inplace=True, axis=1)

data = utils.shuffle(data, random_state=42)

y = np.array(data['LDigit'])
X = data.drop(['WDigit', 'LDigit'], axis=1).values

X = preprocessing.scale(X)

print('Input shape:', X.shape)

rfc = RandomForestClassifier()

y_pred = cross_val_predict(rfc, X, y)
report = classification_report(y, y_pred)
print(report, end='\n\n')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

def run_prob_cv(X, y, clf, roc=False):
    kf = KFold(n_splits=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf.fit(X_train,y_train)
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob

pred_prob = run_prob_cv(X, y, rfc)
pred = pred_prob[:,1]
wins = y == 1

counts = pd.value_counts(pred)

true_prob = defaultdict(float)

for prob in counts.index:
    true_prob[prob] = np.mean(wins[pred == prob])
true_prob = pd.Series(true_prob)

counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
print(counts)
