from __future__ import division

import numpy as np
import pandas as pd
import time

from pandas import Series, DataFrame
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing

def generate_frames():
    """
        """
    
    t0 = time.time()
    print 'Generating data frames...'

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    print train_df.shape
    print test_df.shape

    drop_list = []
    for col in train_df.columns:
        if train_df[col].std() == 0:
            drop_list.append(col)

    train_df = train_df.drop(drop_list, axis=1)
    test_df = test_df.drop(drop_list, axis=1)

    drop_list = []
    c = train_df.columns
    for ii in range(len(c) - 1):
        v = train_df[c[ii]].values
        for jj in range(ii+1, len(c)):
            if np.array_equal(v, train_df[c[jj]].values):
                drop_list.append(c[jj])

    train_df = train_df.drop(drop_list, axis=1)
    test_df = test_df.drop(drop_list, axis=1)

    print train_df.shape
    print test_df.shape

    X = train_df.drop(['TARGET'], axis=1)
    y = train_df.TARGET

    print 'Finished generating data frames in: %.2fs\n'%(time.time() - t0)
    
    return train_df, test_df, X, y

if __name__ == '__main__':

    train_df, test_df, X, y = generate_frames()

    final_train_df = train_df.drop(['TARGET', 'ID'], axis=1)
    final_targets_df = train_df.TARGET

    Ids = test_df.ID
    final_test_df = test_df.drop(['ID'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    X_scaler = preprocessing.StandardScaler()
    scaled_X_train = X_scaler.fit_transform(X_train)
    scaled_X_test = X_scaler.transform(X_test)

    clf_args = {'criterion':'entropy', 'bootstrap':True, 'class_weight':'balanced', 'min_samples_split':2}

    pipeline = Pipeline([
                        ('clf', RandomForestClassifier(**clf_args))
                        ])

    parameters = {
                    'clf__n_estimators': (100, 200, 250),
#                    'clf__min_samples_split': np.arange(1, 5, 1),
                    'clf__max_depth': np.arange(7, 10, 1),
                    'clf__min_samples_leaf': np.arange(4, 8, 1)
                 }

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=1, scoring='roc_auc', cv=3)
    grid_search.fit(scaled_X_train, y_train)
    print 'Best score: %.3f'%grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' %(param_name, best_parameters[param_name])

    predictions = grid_search.predict(scaled_X_test)
    print classification_report(y_test, predictions)

    for param_name in parameters.keys():
        clf_args[param_name[5:]] = best_parameters[param_name]

    print 'clf_args:', clf_args

    final_scaler = preprocessing.StandardScaler()
    scaled_final_train_df = final_scaler.fit_transform(final_train_df)
    scaled_final_test_df = final_scaler.transform(final_test_df)

    classifier = RandomForestClassifier(**clf_args)
    classifier.fit(scaled_final_train_df, final_targets_df)
    output = classifier.predict_proba(scaled_final_test_df)
   
    output_probabilities = [round(x[1], 3) for x in output]

    S = Series(output_probabilities, index=Ids)
    S.to_csv('Santander_randomForest_results.csv', header=True, index_label=    ['ID', 'TARGET'])
