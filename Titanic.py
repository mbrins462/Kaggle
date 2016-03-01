#! /usr/bin/env python

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from sklearn.ensemble import RandomForestClassifier

train_file = pd.read_csv('train.csv')
test_file = pd.read_csv('test.csv')

train_df = DataFrame(train_file)
test_df = DataFrame(test_file)

# drop columns that don't have much affect on outcome
train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# turn gender string into an integer using a new column titled
# 'Gender' and drop the 'Sex' column
train_df['Gender'] = train_df.Sex.map({'female': 0, 'male': 1}).astype(int)
train_df = train_df.drop(['Sex'], axis=1)
test_df['Gender'] = test_df.Sex.map({'female': 0, 'male': 1}).astype(int)
test_df = test_df.drop(['Sex'], axis=1)

# passengers 61 and 829 don't have 'Embarked' values
# they are both 1st class women, from the data the mean class of women
# embarking at the different ports are
# 'C' ---> female_mean_class = 1.726
# 'Q' ---> female_mean_class = 2.889
# 'S' ---> female_mean_class = 2.197
# it seems reasonable to guess that these passengers embarked at Cherbourg
# the test data is complete in the 'Embarked' column

train_df.loc[ (train_df.Embarked.isnull()), 'Embarked'] = 'C'
train_df.Embarked = train_df.Embarked.map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
test_df.Embarked = test_df.Embarked.map({'C': 0, 'Q': 1, 'S': 2}).astype(int)

# some passenger ages are missing so let's duplicate the 'Age' column with
# an 'AgeFill' column and then populate the missing values 
train_df['AgeFill'] = train_df.Age
test_df['AgeFill'] = test_df.Age

mean_ages = np.zeros((2, 3, 3))
for ii in range(2):
    for jj in range(3):
        for kk in range(3):
            mean_ages[ii, jj, kk] = train_df[ (train_df['Gender'] == ii) & (train_df['Pclass'] == jj+1) \
                                            & (train_df['Embarked'] == kk)]['Age'].dropna().mean()

for ii in range(2):
    for jj in range(3):
        for kk in range(3):
            train_df.loc[ (train_df.Age.isnull()) & (train_df.Gender == ii) & (train_df.Pclass == jj+1) \
                        & (train_df.Embarked == kk), 'AgeFill'] = mean_ages[ii, jj, kk]
            test_df.loc[ (test_df.Age.isnull()) & (test_df.Gender == ii) & (test_df.Pclass == jj+1) \
                        & (test_df.Embarked == kk), 'AgeFill'] = mean_ages[ii, jj, kk]

train_df = train_df.drop(['Age'], axis=1)
test_df = test_df.drop(['Age'], axis=1)

# It could be that one's gender and class couple together in their effects so we will
# add new column 'GenderClass' which takes the product of passenger gender value (plus one) and class
# examples: a first class female would have GenderClass = (0 + 1) * 1 = 1
#           a second class male would have GenderClass = (1 + 1) * 2 = 4
train_df['GenderClass'] = (train_df.Gender + 1) * train_df.Pclass
test_df['GenderClass'] = (test_df.Gender + 1) * test_df.Pclass

# Family size could play a role in survival rate as well
train_df['FamilySize'] = train_df.SibSp + train_df.Parch
test_df['FamilySize'] = test_df.SibSp + test_df.Parch

# One's gender may also couple to family size when determining survival as well just as
# gender and class (again it will be gender + 1 as was in GenderClass
# we also take family size + 1 in order to distinguish men and women that are alone)

train_df['GenderFamilySize'] = (train_df.Gender + 1) * (train_df.FamilySize + 1)
test_df['GenderFamilySize'] = (test_df.Gender + 1) * (test_df.FamilySize + 1)

# passenger 1044 is a 3rd class male 60.5yrs old who embarked at Southampton
# and he is missing his fare value, from data the mean fare for 3rd class 
# man embarking at Southampton is 13.307149

test_df.loc[ (test_df.Fare.isnull()), 'Fare'] = 13.307149

# We are left with columns for 'Pclass', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Gender', 'AgeFill', 'GenderClass', 'FamilySize', 'GenderFamilySize'

train_data = train_df.values
test_data = test_df.values

# begin the random forest

forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::], train_data[0::, 0])
output = forest.predict(test_data)

PassengerIds = np.arange(892, 1310)
S = Series(output, index=PassengerIds, dtype=int)
S.to_csv('results.csv', header=True, index_label=['PassengerId','Survived'])

