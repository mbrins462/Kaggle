import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.neighbors import KNeighborsClassifier

train_file = pd.read_csv('train.csv')
test_file = pd.read_csv('test.csv')

train_df = DataFrame(train_file)
test_df = DataFrame(test_file)

# make separate data frame for digits and take out of training set
target_df = train_df.label
train_df = train_df.drop(['label'], axis=1)

train_data = train_df.values.astype(np.uint8)
target_data = target_df.values.astype(np.uint8)
test_data = test_df.values.astype(np.uint8)

n_neighbors, weights = 20, 'distance'

clf = KNeighborsClassifier(n_neighbors, weights=weights)
clf.fit(train_data, target_data)

print 'Starting k-neighbors...'

output = clf.predict(test_data)

ImageIds = np.arange(1, 28001)
S = Series(output, index=ImageIds, dtype=np.uint8)
S.to_csv('kNeighbors_results.csv', header=True, index_label=['ImageId', 'Label'])
