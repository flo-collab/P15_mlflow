import pandas as pd



def get_data(path='../01-Kaggle-Taxi-Fare/data/train.csv', nrows=10000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(path, sep=',', nrows=nrows)
    return df


def clean_data(df, test=False):
    df = df[
        (df.fare_amount > 0) &
        (df.passenger_count <= 8) &
        (df.passenger_count > 0)&
        (df["pickup_latitude"].between(left = 40, right = 42 ))&
        (df["pickup_longitude"].between(left = -74.3, right = -72.9 ))&
        (df["dropoff_latitude"].between(left = 40, right = 42 ))&
        (df["dropoff_longitude"].between(left = -74, right = -72.9 ))

        ]
    return df

def get_and_clean():
    return



# def get_data(path='../01-Kaggle-Taxi-Fare/data/train.csv', nrows=10000):
#     '''returns a DataFrame with nrows from s3 bucket'''
#     df = pd.read_csv(path, sep=',', nrows=nrows)
#     return df