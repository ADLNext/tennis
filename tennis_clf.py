import numpy as np
import pandas as pd

from sklearn import svm, tree
from sklearn import preprocessing, utils
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

data = pd.read_csv('data/match_matrix.csv', delimiter=', ', engine='python').dropna(axis=0)

'''
le = preprocessing.LabelEncoder()
to_encode = ['Court', 'Surface', 'Round', 'Series']
for col in to_encode:
    le.fit(data[col])
    data[col] = le.transform(data[col])
'''

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

models = {
    'Support Vector Machine (Linear)': svm.SVC(kernel='linear'),
    'AdaBoost': AdaBoostClassifier(),
    'Decision Tree': tree.DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    '(16, 16, 16, 32, 16, 16)': MLPClassifier(
        solver='lbfgs',
        activation="relu",
        alpha=1e-5,
        hidden_layer_sizes=(16, 16, 16, 32, 16, 16),
        random_state = 1),
    '(16, 16, 32, 64, 32, 16)': MLPClassifier(
        solver='lbfgs',
        activation="relu",
        alpha=1e-5,
        hidden_layer_sizes=(16, 16, 32, 64, 32, 16),
        random_state = 1),
    '(16, 16, 32, 64, 64, 32, 16)': MLPClassifier(
        solver='lbfgs',
        activation="relu",
        alpha=1e-5,
        hidden_layer_sizes=(16, 16, 32, 64, 64, 32, 16),
        random_state = 1)
}

for name, model in models.items():
    y_pred = cross_val_predict(model, X, y)
    report = classification_report(y, y_pred)
    print(name)
    print(report, end='\n\n')
