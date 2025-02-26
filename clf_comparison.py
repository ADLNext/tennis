import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn import svm
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

from sklearn.metrics import classification_report, confusion_matrix
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
    'Gaussian Process Classifier': GaussianProcessClassifier(),
    'Support Vector Machine (Linear)': svm.SVC(kernel='linear'),
    'Support Vector Machine (RBF)': svm.SVC(gamma=2, C=1.0),
    'MLP(64, 64)': MLPClassifier(solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(64, 64),
        random_state = 1),
    'MLP(64, 128)': MLPClassifier(solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(64, 128),
        random_state = 1),
    'MLP(128, 128)': MLPClassifier(solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(128, 128),
        random_state = 1),
    'MLP(128, 256)': MLPClassifier(solver='lbfgs',
        alpha=1e-5,
        hidden_layer_sizes=(128, 256),
        random_state = 1),
    'Decision Tree': tree.DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    '5-neighbors': KNeighborsClassifier(5),
    'Naive Bayes': GaussianNB()
}

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

for name, model in models.items():
    y_pred = cross_val_predict(model, X_test, y_test)
    report = classification_report(y_test, y_pred)
    print(name)
    print(report, end='\n\n')

# let's try something deeper, just for fun
y_train_DL = []
for label in y_train:
    if label == 0:
        y_train_DL.append([1, 0])
    else:
        y_train_DL.append([0, 1])

model = Sequential()
model.add(Dense(128, input_dim=X.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
model.fit(
    X_train.as_matrix(), y_train_DL,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)
preds = model.predict_classes(X_test.as_matrix(), verbose=0)
report = classification_report(y_test, preds)
print('Deep NN (cross entropy + rmsprop)')
print(report, end='\n\n')

model.reset_states() 
model.compile(loss='mean_squared_error', optimizer='sgd')
model.fit(
    X_train.as_matrix(), y_train_DL,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=0
)
preds = model.predict_classes(X_test.as_matrix(), verbose=0)
report = classification_report(y_test, preds)
print('Deep NN (mean squared error + sgd)')
print(report, end='\n\n')
