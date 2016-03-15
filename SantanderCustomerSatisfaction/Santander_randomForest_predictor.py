from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, datetime

from pandas import Series, DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA

def generate_data_frame():
    """
        """
    t0 = time.time()
    print 'Generating data frame...'

    train_df = pd.read_csv('train.csv')
    train_df = train_df.drop(['ID'], axis=1)
    
    test_df = pd.read_csv('test.csv')
    Ids = test_df.ID
    test_df = test_df.drop(['ID'], axis=1)

    print 'Finished generating data frame.'
    print 'function to generate data frame took %.1fs\n'%(time.time() - t0)

    return train_df, test_df, Ids

def split_data(df):
    """
        splits training data for cross validation

        """

    X = df.drop(['TARGET'], axis=1)
    y = df.TARGET

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    
    T = time.time()
    
    # load the prepared data frames
    train_df, test_df, Ids = generate_data_frame()
    final_train_df = train_df.drop(['TARGET'], axis=1)
    targets_df = train_df.TARGET

    # split the training data frame into parts for cross validation
    X_train, X_test, y_train, y_test = split_data(train_df)

    # create a pipeline for grid search
    pipeline = Pipeline([
#                       ('pca', PCA()),
                        ('clf', RandomForestClassifier(criterion='entropy', bootstrap=True, class_weight='balanced', min_samples_split=11, max_depth=8, min_samples_leaf=1))
                        ])

    parameters = {
#                  'pca__n_components': (290, 295),
                  'clf__n_estimators': (10, 50, 100)
#                  'clf__min_samples_split': np.arange(10, 13, 1),
#                  'clf__max_depth': np.arange(5, 9, 1)
                  }

    # begin the grid search
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=4, verbose=1, scoring='roc_auc', cv=5)
    grid_search.fit(X_train, y_train)
    print 'Best score: %0.3f'%grid_search.best_score_
    print 'Best parameters set:'
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print '\t%s: %r' %(param_name, best_parameters[param_name])
        if param_name == 'clf__min_samples_split':
            min_samples_split = best_parameters[param_name]
        elif param_name == 'clf__n_estimators':
            n_estimators = best_parameters[param_name]
#        elif param_name == 'pca__n_components':
#            n_components = best_parameters[param_name]
        elif param_name == 'clf__max_depth':
            max_depth = best_parameters[param_name]


    predictions = grid_search.predict(X_test)
    print classification_report(y_test, predictions)

#    pca = PCA(n_components=n_components)

#    train_df_reduced = pca.fit_transform(train_df)
#    test_df_reduced = pca.transform(test_df)

    classifier = RandomForestClassifier(criterion='entropy', bootstrap=True, class_weight='balanced', n_estimators=n_estimators, max_depth=8, min_samples_split=11, min_samples_leaf=1)
    classifier.fit(final_train_df, targets_df)
    output = classifier.predict(test_df)

    S = Series(output, index=Ids, dtype=int)
    S.to_csv('Santander_randomForest_results.csv', header=True, index_label=['ID', 'TARGET'])

