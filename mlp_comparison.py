import os
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, cross_val_predict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

df = pd.concat([
    pd.read_excel('data/matches/2017.xlsx'),
    pd.read_excel('data/matches/2016.xlsx'),
    pd.read_excel('data/matches/2015.xlsx'),
    pd.read_excel('data/matches/2014.xlsx'),
    pd.read_excel('data/matches/2013.xlsx')
], axis = 0)

history = pd.concat([
    df[(df['Winner'] == 'Djokovic N.') | (df['Winner'] == 'Nadal R.')],
    df[(df['Loser'] == 'Djokovic N.') | (df['Loser'] == 'Nadal R.')]
])

df_player1 = history.head(len(history)//2)
df_player2 = history.tail(len(history) - len(df_player1))

final_1 = pd.DataFrame()
final_1['RankDiff'] = df_player1['WRank'] - df_player1['LRank']
final_1['y'] = 1
final_2 = pd.DataFrame()
final_2['RankDiff'] = df_player2['LRank'] - df_player2['WRank']
final_2['y'] = 0

final = pd.concat([
    final_1,
    final_2
])

final = pd.concat([
    final,
    pd.get_dummies(history['Surface']),
    pd.get_dummies(history['Court']),
    pd.get_dummies(history['Location']),
    pd.get_dummies(history['Tournament'])
], axis=1)

elapsed = pd.concat([
    df_player1['Date'] - df_player1['Date'].shift(1),
    df_player2['Date'] - df_player2['Date'].shift(1)
])

final = pd.concat([
    final,
    elapsed
], axis=1)

final = final[~final['Date'].isnull()]
final['Elapsed'] = final['Date'].apply(lambda x: np.absolute(x.days))
final.drop('Date', axis=1, inplace=True)

y = final['y']
X = final.drop(['y'], axis=1)

models = {
    '(32, 32, 32)': MLPClassifier(solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(32, 32, 32),
        random_state = 1),
    '(64, 64, 64)': MLPClassifier(solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(64, 64, 64),
        random_state = 1),
    '(64, 128, 64)': MLPClassifier(solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(64, 128, 64),
        random_state = 1),
    '(64, 128, 128)': MLPClassifier(solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(64, 128, 128),
        random_state = 1),
    '(64, 64, 64, 64)': MLPClassifier(solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(64, 64, 64, 64),
        random_state = 1),
    '(128, 128)': MLPClassifier(solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(128, 128),
        random_state = 1)
}

for name, model in models.items():
    y_pred = cross_val_predict(model, X, y)
    report = classification_report(y, y_pred)
    print(name)
    print(report, end='\n\n')
