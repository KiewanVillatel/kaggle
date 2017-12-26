import os

import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sacred import Experiment
from sacred.observers import MongoObserver
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras import regularizers
from tensorflow import set_random_seed

ex = Experiment()


ex.observers.append(MongoObserver.create(os.environ['mongodb_url'], db_name=os.environ['mongodb_db']))

@ex.config
def my_config():
  hidden_size = 128
  nb_models = 1
  dropout_rate = 0
  kernel_regularizer_rate = 0.001
  batch_size = 32
  epochs = 1000
  learning_rate = 0.00001
  decay = 1e-7
  validation_split = 0.1

  fare_modulo = 5
  age_modulo = 5

  model_type='mlp'


ex.add_config('config.yaml')

LABEL = 'Survived'
ID = 'PassengerId'
PREDICTION = 'Prediction'


@ex.capture
def build_datasets(fare_modulo, age_modulo, seed):
  df_train = pd.read_csv('train.csv')
  df_train = df_train.sample(frac=1, random_state=seed)
  df_test = pd.read_csv('test.csv')
  df = pd.concat([df_train, df_test])

  FEATURES = ['Pclass', 'Sex', 'AgeModulo', 'SibSp', 'Parch', 'Embarked', 'FareModulo',
              'IsBaby', 'IsYoungChildren', 'IsChildren', 'IsAdolescent', 'IsYoungAdult', 'IsAdult', 'IsOld', 'IsVeryOld',
              'IsInCabin']

  df['AgeModulo'] = df['Age'] % age_modulo
  df['IsBaby'] = df['Age'].apply(lambda x: x < 2)
  df['IsYoungChildren'] = df['Age'].apply(lambda x: 2 <= x < 7)
  df['IsChildren'] = df['Age'].apply(lambda x: 7 <= x < 13)
  df['IsAdolescent'] = df['Age'].apply(lambda x: 13 <= x < 18)
  df['IsYoungAdult'] = df['Age'].apply(lambda x: 18 <= x < 30)
  df['IsAdult'] = df['Age'].apply(lambda x: 30 <= x < 50)
  df['IsOld'] = df['Age'].apply(lambda x: 50 <= x < 70)
  df['IsVeryOld'] = df['Age'].apply(lambda x: x >= 70)
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
def build_mlp_model(mlp_input_size, hidden_size, dropout_rate, learning_rate, decay, kernel_regularizer_rate):
  model = Sequential()
  model.add(Dense(units=hidden_size, activation='relu', input_dim=mlp_input_size, kernel_regularizer=regularizers.l2(kernel_regularizer_rate)))
  model.add(Dropout(dropout_rate, noise_shape=None, seed=None))
  model.add(Dense(units=hidden_size, activation='relu', input_dim=hidden_size, kernel_regularizer=regularizers.l2(kernel_regularizer_rate)))
  model.add(Dropout(dropout_rate, noise_shape=None, seed=None))
  model.add(Dense(units=2, activation='softmax', kernel_regularizer=regularizers.l2(kernel_regularizer_rate)))
  model.compile(loss='binary_crossentropy',
                optimizer=optimizers.Adam(lr=learning_rate, decay=decay),
                metrics=['accuracy'])
  return model


@ex.capture
def build_random_forest_model():
  model = RandomForestClassifier()
  return model


@ex.capture
def build_model(mlp_input_size, model_type):
  if model_type == 'mlp':
    return build_mlp_model(mlp_input_size=mlp_input_size)
  elif model_type == 'rf':
    return build_random_forest_model()


@ex.capture
def fit_model(model, x, y, model_type, batch_size, epochs, validation_split):
  if model_type == 'mlp':
    model.fit(x,
              y,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=validation_split,
              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')],
              verbose=0)
  elif model_type == 'rf':
    model.fit(x, y)


@ex.automain
def my_main(_run, nb_models, seed):
  set_random_seed(seed)
  np.random.seed(seed)
  one_hot_features, df_train, df_valid, df_test = build_datasets()

  df_test[LABEL] = 0
  df_valid[PREDICTION] = 0

  for i in range(nb_models):
    model = build_model(mlp_input_size=len(one_hot_features))
    fit_model(model, df_train[one_hot_features], pd.get_dummies(df_train[LABEL]))
    df_test[LABEL] += model.predict(df_test[one_hot_features])[:, 1]
    df_valid[PREDICTION] += model.predict(df_valid[one_hot_features])[:, 1]

  df_test[LABEL] /= nb_models
  df_test[LABEL] = np.rint(df_test[LABEL]).astype(int)

  df_valid[PREDICTION] /= nb_models
  df_valid[PREDICTION] = np.rint(df_valid[PREDICTION])

  df_test[[ID, LABEL]].to_csv(path_or_buf='./out.csv', index=False)

  return accuracy_score(df_valid[LABEL], df_valid[PREDICTION])
