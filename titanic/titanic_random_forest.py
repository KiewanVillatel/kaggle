import os

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

ex = Experiment()
ex.observers.append(MongoObserver.create(os.environ['mongodb_url'], db_name=os.environ['mongodb_db']))

@ex.config
def my_config():
  nb_models = 1
  fare_modulo = 5
  age_modulo = 5

ex.add_config('config_random_forest.json')

LABEL = 'Survived'
ID = 'PassengerId'
PREDICTION = 'Prediction'

@ex.capture
def build_datasets(fare_modulo, age_modulo):
  df_train = pd.read_csv('train.csv')
  df_train = df_train.sample(frac=1)
  df_test = pd.read_csv('test.csv')
  df = pd.concat([df_train, df_test])

  FEATURES = ['Pclass', 'Sex', 'AgeModulo', 'SibSp', 'Parch', 'Embarked', 'FareModulo',
              'IsBasy', 'IsYoungChildren', 'IsChildren', 'IsAdolescent', 'IsOld',
              'IsInCabin']

  df['AgeModulo'] = df['Age'] % age_modulo
  df['IsBasy'] = df['Age'] < 2
  df['IsYoungChildren'] = df['Age'] < 7
  df['IsChildren'] = df['Age'] < 12
  df['IsAdolescent'] = df['Age'] < 18
  df['IsOld'] = df['Age'] > 60
  df['FareModulo'] = df['Fare'] % fare_modulo
  df['IsInCabin'] = pd.isnull(df['Cabin']).apply(lambda x: not x)

  # Convert to categorical
  df[LABEL] = df[LABEL].astype(str)
  df[FEATURES] = df[FEATURES].astype(str)
  df_one_hot = pd.get_dummies(df[FEATURES], dummy_na=False)
  one_hot_features = df_one_hot.columns.values
  df_one_hot[LABEL] = df[LABEL]
  df_one_hot[ID] = df[ID]

  df_one_hot_train = df_one_hot[0:len(df_train.index)]
  df_one_hot_test = df_one_hot[len(df_train.index):]

  df_one_hot_train[LABEL] = df_train[LABEL]
  msk = np.random.rand(len(df_one_hot_train)) < 0.9
  df_one_hot_valid = df_one_hot_train[~msk]
  df_one_hot_train = df_one_hot_train[msk]

  return one_hot_features, df_one_hot_train, df_one_hot_valid, df_one_hot_test


@ex.capture
def build_model():
  model = RandomForestClassifier()
  return model

@ex.automain
def my_main(nb_models):

  one_hot_features, df_train, df_valid, df_test = build_datasets()

  df_test[LABEL] = 0
  df_valid[PREDICTION] = 0

  for i in range(nb_models):
    model = build_model()
    model.fit(df_train[one_hot_features],
              pd.get_dummies(df_train[LABEL]))
    df_test[LABEL] += model.predict(df_test[one_hot_features])[:, 1]
    df_valid[PREDICTION] += model.predict(df_valid[one_hot_features])[:, 1]

  df_test[LABEL] /= nb_models
  df_test[LABEL] = np.rint(df_test[LABEL]).astype(int)

  df_valid[PREDICTION] /= nb_models
  df_valid[PREDICTION] = np.rint(df_valid[PREDICTION])

  df_test[[ID, LABEL]].to_csv(path_or_buf='./out.csv', index=False)

  return accuracy_score(df_valid[LABEL], df_valid[PREDICTION])
