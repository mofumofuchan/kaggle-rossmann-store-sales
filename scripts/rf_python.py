import pandas as pd
import numpy as np

# Import all utility functions
import utils
import gplearn.skutils

stores = pd.read_csv('../input/store.csv')
train = pd.read_csv('../input/train.csv', parse_dates = ['Date'])
test = pd.read_csv('../input/test.csv', parse_dates = ['Date'])


def process(input_data, store_data, max_comp_distance=100000, sort_by=None):
    
    # Create a copy of the data
    data = input_data.copy()
    
    if sort_by:
        data.sort_values(by=sort_by, inplace=True)

    # Merge the Store information to the data
    data = data.merge(store_data, on='Store')
    data.drop(['Store'], axis=1, inplace=True)
    
    if 'Sales' not in data.columns:
        # Merge creates new Ids, so we need to reset the Ids
        # on the Id column for the test set
        data.set_index('Id', inplace=True)    
    
    # Process the Date field
    data['year'] = data.Date.apply(lambda x: x.year)
    data['month'] = data.Date.apply(lambda x: x.month)
    # data['day'] = data.Date.apply(lambda x: x.day)
    data['woy'] = data.Date.apply(lambda x: x.weekofyear)
    data.drop(['Date'], axis = 1, inplace=True)
    
    # Normalize Competition Distance
    data['CompetitionDistance'] = data.CompetitionDistance.fillna(max_comp_distance)
    
    # Process the Competition Open fields
    data['CompetitionOpen'] = 12 * (data.year - data.CompetitionOpenSinceYear) + (data.month - data.CompetitionOpenSinceMonth)
    data['CompetitionOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data.drop(['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], axis=1, inplace=True)
    
    # Process the Promo Open field
    data['PromoOpen'] = 12 * (data.year - data.Promo2SinceYear) + (data.woy - data.Promo2SinceWeek) / float(4)
    data['PromoOpen'] = data.CompetitionOpen.apply(lambda x: x if x > 0 else 0)
    data.drop(['Promo2SinceYear', 'Promo2SinceWeek'], axis=1, inplace=True)
    
    # Normalize State Holiday field
    data['StateHoliday'] = data.StateHoliday.apply(lambda x: x if x in ['a', 'b', 'c'] else 0)
    
    # Dummy Coding
    for dummy in ['StateHoliday', 'StoreType', 'Assortment', 'DayOfWeek']:
        # Create dummy columns
        data = pd.get_dummies(data, columns=[dummy])
        
        # Remove original column
        if dummy in data.columns:
            data.drop([dummy], axis=1, inplace=True)
    
    # Fix State Holiday columns, some values are not present in the testing data
    for col in ['StateHoliday_0', 'StateHoliday_a', 'StateHoliday_b', 'StateHoliday_c']:
        if col not in data.columns:
            data[col] = np.zeros(len(data.index))
    
    # Drop unused Columns
    data.drop(['PromoInterval'], axis=1, inplace=True)
    
    # Make sure columns are sorted
    data = data.reindex_axis(sorted(data.columns), axis=1)
    
    # training data
    if 'Sales' in data.columns:
        
        # Remove NaN values
        data.fillna(0, inplace=True)
    
        # Consider only open stores for training. Closed stores wont count into the score
        data = data[data.Open != 0]
    
        # Use only Sales bigger then zero
        data = data[data.Sales > 0]

        return data.drop(['Sales', 'Customers'], axis=1), data.Sales
    
    # testing data
    else:
        # Remove NaN values
        # appear only in Open column
        data.fillna(1, inplace=True)
        
        return data,

X_train, y_train = process(train, stores)

X_test, = process(test, stores)

from sklearn.ensemble import RandomForestRegressor

# Classifier Parameters
clf_params = {
  'n_estimators': 20
}

# Random Forest Classifier
clf = RandomForestRegressor(**clf_params)

print("start predicting")

clf.fit(X_train.values, y_train.values)
y_pred = clf.predict(X_test.values)


result = pd.DataFrame({'Id': X_test.index.values, 'Sales': y_pred})
result.to_csv('submission_%s.csv' % utils.timestamp(), index=False)

