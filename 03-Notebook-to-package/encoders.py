import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from utils import haversine_vectorized


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self,time_col ='pickup_datetime', timezone = 'America/New_York'):
       self.time_col = time_col
       self.timezone = timezone


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.time_col] = pd.to_datetime(X_[self.time_col], format="%Y-%m-%d %H:%M:%S %Z").dt.tz_convert(self.timezone)
        X_['dow']=X_[self.time_col].dt.dayofweek
        X_['hour'] = X_[self.time_col].dt.hour
        X_['month'] = X_[self.time_col].dt.month
        X_['year'] = X_[self.time_col].dt.year
        X_ = X_.set_index(self.time_col)
        
        return X_[['dow', 'hour']]



class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        X_ = X.copy()
        X_["distance"] = haversine_vectorized(X_, 
                                    self.start_lat, self.start_lon,
                                    self.end_lat, self.end_lon)
        return X_[['distance']]



