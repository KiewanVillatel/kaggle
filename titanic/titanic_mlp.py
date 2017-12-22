import os

import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
ex.observers.append(MongoObserver.create(os.environ['mongodb_url'], db_name=os.environ['mongodb_db']))

@ex.config
def my_config():
  hidden_size = 128
  nb_models = 1
  dropout_rate = 0


@ex.capture
def build_model(hidden_size, input_size, dropout_rate):
  model = Sequential()
  model.add(Dense(units=hidden_size, activation='relu', input_dim=input_size))
  model.add(Dropout(dropout_rate, noise_shape=None, seed=None))
  model.add(Dense(units=hidden_size, activation='relu', input_dim=hidden_size))
  model.add(Dropout(dropout_rate, noise_shape=None, seed=None))
  model.add(Dense(units=2, activation='softmax'))
  model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(lr=0.00001, decay=1e-7),
                metrics=['accuracy'])
  return model

@ex.automain
def my_main(nb_models):
  df_train = pd.read_csv('train.csv')
  df_train = df_train.sample(frac=1)
  df_test = pd.read_csv('test.csv')
  df = pd.concat([df_train, df_test])

  LABEL = 'Survived'
  ID = 'PassengerId'
  FEATURES = ['Pclass', 'Sex', 'AgeModulo', 'SibSp', 'Parch', 'Embarked', 'FareModulo',
              'IsBasy', 'IsYoungChildren', 'IsChildren', 'IsAdolescent', 'IsOld',
              'IsInCabin']

  df['AgeModulo'] = df['Age'] % 5
  df['IsBasy'] = df['Age'] < 2
  df['IsYoungChildren'] = df['Age'] < 7
  df['IsChildren'] = df['Age'] < 12
  df['IsAdolescent'] = df['Age'] < 18
  df['IsOld'] = df['Age'] > 60
  df['FareModulo'] = df['Fare'] % 5
  df['IsInCabin'] = pd.isnull(df['Cabin']).apply(lambda x: not x)

  # Convert to categorical
  df[LABEL] = df[LABEL].astype(str)
  df[FEATURES] = df[FEATURES].astype(str)
  df_one_hot = pd.get_dummies(df[FEATURES], dummy_na=True)

  df_one_hot_train = df_one_hot[0:len(df_train.index)]
  df_one_hot_test = df_one_hot[len(df_train.index):]

  df_test[LABEL] = 0

  for i in range(nb_models):
    model = build_model(input_size=len(df_one_hot.columns))
    model.fit(df_one_hot_train,
              pd.get_dummies(df_train[LABEL].astype(str)),
              epochs=1000,
              batch_size=32,
              validation_split=0.1,
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')])
    df_test[LABEL] += model.predict(df_one_hot_test)[:, 1]

  df_test[LABEL] /= nb_models
  df_test[LABEL] = np.rint(df_test[LABEL])

  df_test[[ID,LABEL]].to_csv(path_or_buf='./out.csv', index=False)
