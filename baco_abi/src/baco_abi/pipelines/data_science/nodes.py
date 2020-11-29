# Copyright 2020 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code for the nodes in the example pipeline. This code is meant
just for illustrating basic Kedro features.

Delete this when you start working on your own Kedro project.
"""
# pylint: disable=invalid-name

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, plot_roc_curve, accuracy_score
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

def train_rfc(train_x, train_y, test_x, test_y, df_c, df_test):
    clf = RandomForestClassifier()
    clf.fit(train_x, train_y)

    print(clf.score(train_x, train_y))
    print(clf.score(test_x, test_y))
    print(roc_auc_score(test_y, clf.predict_proba(test_x)[:, 1]))

    df_c["predicted"] = clf.predict_proba(df_c.drop(columns=["Prod1"]))[:, 1]

    df_a = df_test.copy().set_index("Cliente")
    df_a["Marca1"] = df_c[["predicted"]]
    df_a

    return df_a


def random_search_rfc(train_x, train_y, test_x, test_y):

    clf = RandomForestClassifier()
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    rf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(train_x, train_y)

    import copy

    best_rf = copy.deepcopy(rf_random.best_estimator_)

    print(best_rf.score(train_x, train_y))
    print(best_rf.score(test_x, test_y))

    print(roc_auc_score(test_y, best_rf.predict_proba(test_x)[:, 1]))



def train_svmc(train_x, train_y, test_x, test_y):
    clf2 = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
    clf2.fit(train_x, train_y)

    print(clf2.score(train_x, train_y))
    print(clf2.score(test_x, test_y))
    print(roc_auc_score(test_y, clf2.predict_proba(test_x)[:, 1]))


def train_xgboost(train_x, train_y, test_x, test_y, data_c):

    param = {"booster":"gbtree", "max_depth": 2, "eta": 0.3, "objective": "binary:logistic", "nthread":2}
    num_round = 100
    train_mat = xgb.DMatrix(train_x, train_y)
    test_mat = xgb.DMatrix(test_x, label=test_y)
    all_mat = xgb.DMatrix(data_c.drop(columns=["Prod1"]), label=data_c[["Prod1"]])

    evaluation = [(test_mat, "eval"), (train_mat, "train")]

    bst = xgb.train(param, train_mat, num_round, evaluation)

    clf3 = xgb.XGBModel(**param)
    clf3.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)], eval_metric='logloss')
    print(roc_auc_score(test_y, bst.predict(test_mat)))
    print(roc_auc_score(test_y, clf3.predict(test_x)))
