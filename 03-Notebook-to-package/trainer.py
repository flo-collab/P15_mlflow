from utils import compute_rmse
from encoders import *
from data import *
from utils import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def f_set_pipeline():
    dist_pipe = Pipeline([('dist_transformer',DistanceTransformer()),('std_scaler',StandardScaler())])
    time_pipe = Pipeline([('time_encoder',TimeFeaturesEncoder()),('one_hot',OneHotEncoder(handle_unknown="ignore"))])

    Preprocessor = ColumnTransformer([
        ('time_pipe',time_pipe,['pickup_datetime']),
        ('dist_pipe',dist_pipe,['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude'])
        ])

    final_pipe = Pipeline([
        ('preprocessing',Preprocessor),
        ('linear_regression',LinearRegression())
        ])
    
    return final_pipe




data_path = '../01-Kaggle-Taxi-Fare/data/'

class Trainer_V1():
    def __init__(self):
        self.df = clean_data(get_data())

    def set_pipeline(self):
        pipeline = f_set_pipeline()
        return pipeline

    def run(self,X_train,y_train, pipeline):
        pipeline.fit(X_train,y_train)
        return pipeline

    def evaluate(self,X_test,y_test,pipeline):
        y_pred = pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse


print('lala')

# class Trainer_V2():
#     def __init__(self, path = '../01-Kaggle-Taxi-Fare/data/train.csv', nrows=10000):
#         self.path = path
#         self.nrows = nrows
#         self.df = clean_data(get_data())

#     def set_pipeline(self):
#         pipeline = f_set_pipeline()
#         return pipeline

#     def run(self,X_train,y_train, pipeline):
#         pipeline.fit(X_train,y_train)
#         return pipeline

#     def evaluate(self,X_test,y_test,pipeline):
#         y_pred = pipeline.predict(X_test)
#         rmse = compute_rmse(y_pred, y_test)
#         return rmse




if __name__ == "__main__":
    print('lala')


    trainer1 = Trainer_V1()
    df = trainer1.df
    X = df.drop("fare_amount", axis=1)
    y = df["fare_amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    pipe1 = trainer1.set_pipeline()
    trained_pipe = trainer1.run(X_train,y_train, pipe1)
    rmse  = trainer1.evaluate(X_test,y_test,trained_pipe)
    print('RMSE = ', rmse)






































# class Trainer():
#     def __init__(self,path='../01-Kaggle-Taxi-Fare/data/train.csv', nrows=10000):
#         df = get_data(path,nrows)
#         df = clean_data(df)
        



#         self
#         return

#     def set_pipeline():
#         dist_pipe = Pipeline([('dist_transformer',DistanceTransformer()),('std_scaler',StandardScaler())])
#         time_pipe = Pipeline([('time_encoder',TimeFeaturesEncoder()),('one_hot',OneHotEncoder(handle_unknown="ignore"))])

#         Preprocessor = ColumnTransformer([
#             ('time_pipe',time_pipe,['pickup_datetime']),
#             ('dist_pipe',dist_pipe,['pickup_longitude', 'pickup_latitude','dropoff_longitude', 'dropoff_latitude'])
#             ])

#         pipeline = Pipeline([
#             ('preprocessing',Preprocessor),
#             ('linear_regression',LinearRegression())
#             ])

#         return pipeline



#     def run(X_train, y_train, pipeline):
#         pipeline.fit(X_train,y_train)
#         return pipeline



#     def evaluate(self,X_test,y_test,pipeline):
#         y_pred = pipeline(X_test)
#         rmse = compute_rmse(y_pred, y_test)
#         return rmse
