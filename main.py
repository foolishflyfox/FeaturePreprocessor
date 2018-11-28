from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class FeatureSelect(BaseEstimator, TransformerMixin):
    def __init__(self, select_type='numeric', max_nan_rate=0.4, drop_columns=[]):
        self.__fitted = False
        self.__select_type = select_type
        # indicate the max missing rate, the columns with 
        # missing rate greater than this will be deleted
        self.__max_nan_rate = max_nan_rate
        # record the columns should deleted setted by user
        self.__origin_drops = set(drop_columns)
        
        self.__numeric_indices = []
        self.__string_indices = []
    
    def fit(self, X, y=None):
        self.__fitted = True
        # delete columns which null rate greater than max_nan_rate
        X_len = float(X.shape[0])
        self.__numeric_indices.clear()
        self.__string_indices.clear()
        for i in range(X.shape[1]):
            if i in self.__origin_drops: continue
            try:
                col = X[:, i].astype('float')
                nan_count = np.sum(np.isnan(col)) 
                if nan_count/X_len < self.__max_nan_rate:
                    self.__numeric_indices.append(i)
            except ValueError:
                col = X[:, i].astype('U')
                nan_count = np.sum(col=='nan')
                if nan_count/X_len < self.__max_nan_rate:
                    self.__string_indices.append(i)
        return self
    
    def transform(self, X, y=None):
        assert self.__fitted, 'please fit before transform'
        if self.__select_type=='numeric':
            return X[:, self.__numeric_indices].astype('float')
        else:
            return X[:, self.__string_indices].astype('U')

class StringFeatureProcess(BaseEstimator, TransformerMixin):
    def __init__(self, max_onehot_size=3):
        self.__max_onehot_size = max_onehot_size
        self.__map_list = []
    def fit(self, X, y=None):
        self.__map_list.clear()
        for i in range(X.shape[1]):
            col = X[:, i]
            values = np.unique(col)
            if len(values) <= self.__max_onehot_size:
                self.__map_list.append({v:i for i, v in enumerate(values)})
            else:
                assert y is not None, 'StringFeatureProcess.fit : Please provide with y'
                cur_dict = dict()
                for v in values:
                    tp_indices = (X[:, i]==v)
                    tp_mean = y[tp_indices].mean()
                    cur_dict[v] = tp_mean
                self.__map_list.append(cur_dict)
        return self
        
    def transform(self, X, y=None):
        new_arrays = []
        for p, cur_map in enumerate(self.__map_list):
            if len(cur_map) <= self.__max_onehot_size:
                cur_array = np.zeros((X.shape[0], len(cur_map)))
                for v, i in cur_map.items():
                    cur_array[X[:, p]==v, i] = 1
            else:
                cur_array = np.zeros((X.shape[0], 1))
                for v, i in cur_map.items():
                    cur_array[X[:, p]==v, 0] = i
                cur_array = cur_array-cur_array.mean()
                if cur_array.std()!=0:
                    cur_array = cur_array/cur_array.std()
            new_arrays.append(cur_array)
        return np.hstack(new_arrays)


def demo():
    from sklearn.pipeline import Pipeline, FeatureUnion
    from sklearn.preprocessing import StandardScaler
    # scikit-0.22 should use SimpleImputer instead of Imputer
    from sklearn.impute import SimpleImputer
    import pandas as pd
    import numpy as np

    train_pd = pd.read_csv('train.csv')
    test_pd = pd.read_csv('test.csv')
    numeric_pipe = Pipeline([
        ('selector', FeatureSelect(drop_columns=[0])),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    string_pipe = Pipeline([
        ('selector', FeatureSelect('string', drop_columns=[0])),
        ('encoder', StringFeatureProcess())
    ])
    full_pipe = FeatureUnion([
        ('numeric', numeric_pipe),
        ('string', string_pipe)
    ])
    train_y = train_pd.values[:, -1]
    train_x = full_pipe.fit_transform(train_pd.values[:, :-1], train_y)
    test_x = full_pipe.transform(test_pd.values)
    # then you can use the data to train your model
