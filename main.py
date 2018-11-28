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




