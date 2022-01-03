def extract_time_features(df):
    df['key'] = pd.to_datetime(df['key'], format="%Y-%m-%d %H:%M:%S.%f").dt.tz_localize(tz='UTC').dt.tz_convert('America/New_York')
    df['hour'] = df['key'].dt.hour
    df['dow']=df['key'].dt.dayofweek
    return df


df = df[df["pickup_latitude"].between(left = 40, right = 42 )]
df = df[df["pickup_longitude"].between(left = -74.3, right = -72.9 )]
df = df[df["dropoff_latitude"].between(left = 40, right = 42 )]
df = df[df["dropoff_longitude"].between(left = -74, right = -72.9 )]


import numpy as np
def haversine_distance(df,
                       start_lat="start_lat",
                       start_lon="start_lon",
                       end_lat="end_lat",
                       end_lon="end_lon"):
    R = 6373.0
    lat1 = df[start_lat].apply(np.radians)
    lon1 = df[start_lon].apply(np.radians)
    lat2 = df[end_lat].apply(np.radians)
    lon2 = df[end_lon].apply(np.radians)

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance




df["distance"] = haversine_distance(df, 
                                    start_lat="pickup_latitude", start_lon="pickup_longitude",
                                    end_lat="dropoff_latitude", end_lon="dropoff_longitude")


def clean_data(df, test=False):
    df = df[
        (df.fare_amount > 0) &
        (df.distance < 100) &
        (df.passenger_count <= 8) &
        (df.passenger_count > 0)
        ]
    return df

df_cleaned = clean_data(df)
"% data removed", (1 - len(df_cleaned) / len(df)) * 100

def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())





# On separe le jeu train en train / val:
from sklearn.model_selection import train_test_split
df_train, df_val = train_test_split(df, test_size=0.1)
df_train = clean_data(df_train)

# Separation des colones
target = "fare_amount"
features = ["distance", "hour", "dow", "passenger_count"]
categorical_features = ["hour", "dow"]


# pd.get_dummies(df, columns=['dow','hour'])

def transform_features(_df, dummy_features=None):
    _df = pd.get_dummies(_df[features], columns=dummy_features)
    dummy_features = _df.columns
    return _df, dummy_features

# Import des données
df = pd.read_csv('./data/train.csv',sep=',',nrows=1_000_010)
df = df.dropna()


    # model training
cat_features = ["hour", "dow"]
from sklearn.linear_model import LassoCV
model = LassoCV(cv=5, n_alphas=5)
X_train, dummy_features = transform_features(df_train, cat_features)
X_train = X_train[dummy_features]
y_train = df_train.fare_amount
model.fit(X_train, y_train)

# Ne marche pas : 

-------
df_train = clean_data(df_train)
X_test, dummy_features = transform_features(df_train, cat_features)
X_test = X_test[dummy_features]
df_test["y_pred"] = model.predict(X_test)
-------




def preprocessing(df):
    df = df.dropna()
    df = clean_data(df)
    df["distance"] = haversine_distance(df, 
                                    start_lat="pickup_latitude", start_lon="pickup_longitude",
                                    end_lat="dropoff_latitude", end_lon="dropoff_longitude")

    target = "fare_amount"
    features = ["distance", "hour", "dow", "passenger_count"]
    categorical_features = ["hour", "dow"]
    
    X, columns_X  = transform_features(df[features],categorical_features)
    y = df.fare_amount

    return X, y

