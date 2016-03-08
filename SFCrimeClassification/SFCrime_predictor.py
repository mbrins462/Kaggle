import numpy as np
import pandas as pd
import time, datetime

from pandas import Series, DataFrame
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

def generate_data_frames():
    """
        """
    t0 = time.time()
    print 'Generating data frames...'

    train_df = pd.read_csv('train.csv', parse_dates=['Dates'])
    test_df = pd.read_csv('test.csv', parse_dates=['Dates'])

    # convert crime categories to numerical values
    categories = list(np.unique(train_df.Category.values))
    train_df.Category = train_df.Category.map({category: ii for (ii, category) in enumerate(categories)})

    # convert dates from strings to numbers
    train_df['Year'] = train_df.Dates.map(lambda x: x.year)
    train_df['Week'] = train_df.Dates.map(lambda x: x.week)

    test_df['Year'] = test_df.Dates.map(lambda x: x.year)
    test_df['Week'] = test_df.Dates.map(lambda x: x.week)

    # create columns for each day of week with binary values
    days = list(np.unique(train_df.DayOfWeek))

    for day in days:
        train_df[day] = 0
        train_df.loc[(train_df.DayOfWeek == day), day] = 1

        test_df[day] = 0
        test_df.loc[(test_df.DayOfWeek == day), day] = 1

    # create columns for each police district with binary values
    districts = list(np.unique(train_df.PdDistrict))

    for district in districts:
        train_df[district] = 0
        train_df.loc[(train_df.PdDistrict == district), district] = 1

        test_df[district] = 0
        test_df.loc[(test_df.PdDistrict == district), district] = 1

    train_df = train_df.drop(['PdDistrict', 'DayOfWeek', 'Dates', 'Descript', 'Address', 'Resolution'], axis=1)
    test_df = test_df.drop(['PdDistrict', 'DayOfWeek', 'Dates', 'Address'], axis=1)

    Ids = test_df.Id

    test_df = test_df.drop(['Id'], axis=1)

    print 'Finished generating data frames.\n'
    print 'function to generate data frames took %.1fs'%(time.time() - t0)

    return train_df, test_df, Ids, categories

def generate_index_label(initial, category_list):
    """
        """

    index_label = initial
    for ii in range(len(category_list)):
        index_label = index_label+','+category_list[ii]

    return index_label

if __name__ == '__main__':
    
    train_df, test_df, Ids, categories = generate_data_frames()

    X_train, X_target = train_df.drop(['Category'], axis=1), train_df.Category

    n_neighbors, weights = 100, 'distance'
    classifier = KNeighborsClassifier(n_neighbors, weights=weights)
    
    classifier.fit(X_train, X_target)

    print 'Starting the k-NearestNeighbors Classifier...'
    t0 = time.time()
    output = classifier.predict_proba(test_df)
    print 'Classifier took %.1fs'%(time.time() - t0)

    columns = ['Id']
    for ii in range(len(categories)):
        columns.append(categories[ii])

    D = DataFrame(output, index=Ids)
    D.to_csv('SFCrime_kNeighbors_results.csv', header=True, index_label=generate_index_label('Id', categories))

