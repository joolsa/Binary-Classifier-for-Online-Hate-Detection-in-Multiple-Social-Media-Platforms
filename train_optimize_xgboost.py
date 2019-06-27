"""This file performs a hyperparameter optimization, train the best model on the full train set and reports
 performance on the test set and then retrains the same model on all data (train + test).

It needs the following input:

- data_path: Location of a dataset with the BERT features. These features have to be created with the
create_bert_dataset.py script. Because of the BERT models size it is a slow process to create these features and the
xgboost model will probably retrained with higher frequency than the BERT model, so I separated these two tasks.

- number_of_runs: Number of models that are trained during the hyperparameter optimization.
- optimization_output_path: Location where the results of the hyperparameter optimization are saved.
- final_model_path: Location where the xgboost model is saved.
"""

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, f1_score, roc_curve, roc_auc_score
from sklearn.externals import joblib
from functions import evaluate_model

# --------- User Input ----------------------------
data_path = "data/bert_features.csv"
number_of_runs = 20
optimization_output_path = "optimization_result.csv"
final_model_path = "xgboost_model.dat"
# -------------------------------------------------

# load data
data = pd.read_csv(data_path)
bert_features = ["bert_" + str(i) for i in range(768)]

# validation split
X_train, X_test = train_test_split(data, test_size=0.3)
X_train, X_vali = train_test_split(X_train, test_size=0.3)
print("X_train.shape: {}".format(X_train.shape))
print("X_vali.shape: {}".format(X_vali.shape))
print("X_test.shape: {}".format(X_test.shape))

hyper_list = []
n = 1
for i in range(number_of_runs):
    print("Training model {} out of {}".format(n, number_of_runs))
    learning_rate = np.random.uniform(0.001, 0.15)
    max_depth = np.random.choice([3, 4, 5, 6])
    n_estimators = np.random.randint(low=50, high=180)
    subsample = min(np.random.uniform(0.6, 1.1), 1.0)
    colsample_bytree = min(np.random.uniform(0.6, 1.1), 1.0)

    params = {'learning_rate': learning_rate,
              'max_depth': max_depth,
              'n_estimators': n_estimators,
              'subsample': subsample,
              'colsample_bytree': colsample_bytree}
    print(params)

    xgb_model = xgb.XGBRegressor(learning_rate=learning_rate,
                                 objective='binary:logistic',
                                 random_state=543333,
                                 n_jobs=8,
                                 n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 subsample=subsample,
                                 colsample_bytree=colsample_bytree)

    xgb_model.fit(X_train.loc[:, bert_features], X_train['Hateful_or_not'])

    # evaluation on validation set
    preds = xgb_model.predict(X_vali.loc[:, bert_features])
    fpr, tpr, _ = roc_curve(X_vali['Hateful_or_not'], preds)
    roc_auc = auc(fpr, tpr)
    params['roc'] = roc_auc

    hyper_list.append(pd.DataFrame(params, index=[0]))
    n = n + 1

hyper_df = pd.concat(hyper_list)
hyper_df.sort_values('roc', inplace=True, ascending=False)
hyper_df.reset_index(drop=True, inplace=True)
hyper_df.to_csv(optimization_output_path)

print('Retrain optimal model on full train set.')
X_train = pd.concat([X_train, X_vali])
best_xgb_model = xgb.XGBRegressor(learning_rate=hyper_df['learning_rate'][0],
                                  objective='binary:logistic',
                                  random_state=543333,
                                  n_jobs=8,
                                  n_estimators=hyper_df['n_estimators'][0],
                                  max_depth=hyper_df['max_depth'][0],
                                  subsample=hyper_df['subsample'][0],
                                  colsample_bytree=hyper_df['colsample_bytree'][0])

best_xgb_model.fit(X_train.loc[:, bert_features], X_train['Hateful_or_not'])

preds = best_xgb_model.predict(X_test.loc[:, bert_features])
print('Performance on test set:')
evaluate_model(X_test['Hateful_or_not'], preds)


print('Retrain optimal model on all data.')
X_train = pd.concat([X_train, X_test])
best_xgb_model = xgb.XGBRegressor(learning_rate=hyper_df['learning_rate'][0],
                                  objective='binary:logistic',
                                  random_state=543333,
                                  n_jobs=8,
                                  n_estimators=hyper_df['n_estimators'][0],
                                  max_depth=hyper_df['max_depth'][0],
                                  subsample=hyper_df['subsample'][0],
                                  colsample_bytree=hyper_df['colsample_bytree'][0])
best_xgb_model.fit(X_train.loc[:, bert_features], X_train['Hateful_or_not'])

joblib.dump(best_xgb_model, final_model_path)
