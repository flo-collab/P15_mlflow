from utils import compute_rmse
from encoders import *
from data import *
from utils import *

from joblib import dump, load

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
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










class Trainer_V2():
    def __init__(self, experiment_name
    , path = '../01-Kaggle-Taxi-Fare/data/train.csv', nrows=10000
     ):
        self.experiment_name = experiment_name
        self.path = path
        self.nrows = nrows
        self.df = clean_data(get_data(
            path = self.path, nrows = self.nrows
            ))

    def set_train_test(self, test_size=0.33, random_state=42):
        self.X = self.df.drop("fare_amount", axis=1)
        self.y = self.df["fare_amount"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def set_pipeline(self,):
        self.pipeline = f_set_pipeline()
        print('pipe set')

    def run(self):
        self.pipeline.fit(self.X_train,self.y_train)
        print('pipe trained')


    def evaluate(self):
        self.y_pred = self.pipeline.predict(self.X_test)
        self.rmse = compute_rmse(self.y_pred, self.y_test)
        print('la rmse est :',self.rmse)

        self.mlflow_log_metric('RMSE', self.rmse)
        return self.rmse

    @memoized_property
    def mlflow_client(self):
        # mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self,filename:str):
        dump(self.pipeline, filename+'.joblib')












class Trainer_V3():
    def __init__(self, experiment_name
    , path = '../01-Kaggle-Taxi-Fare/data/train.csv', nrows=10000
     ):
        self.experiment_name = experiment_name
        self.path = path
        self.nrows = nrows
        self.df = clean_data(get_data(
            path = self.path, nrows = self.nrows
            ))

    def set_train_test(self, test_size=0.33, random_state=42):
        self.X = self.df.drop("fare_amount", axis=1)
        self.y = self.df["fare_amount"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

    def set_pipeline(self,estimator,dict_params ={}):
        self.estimator = estimator
        self.dict_params = dict_params
        self.pipeline = f_set_pipeline_custom(self.estimator,dict_params)
        print('pipe set')

    def run(self):
        cv_score = cross_val_score(self.pipeline, self.X, self.y, cv=5).mean()
        cv_rmse = cross_val_score(self.pipeline, self.X, self.y, cv=5,scoring=make_scorer(mean_squared_error)).mean()
        print('cv_score : ',cv_score)
        print('cv_rmse : ',cv_rmse)
        self.mlflow_log_metric('CV score', cv_score)
        self.mlflow_log_metric('CV RMSE', cv_rmse)
        self.pipeline.fit(self.X_train,self.y_train)
        print('pipe trained')


    def evaluate(self):
        self.y_pred = self.pipeline.predict(self.X_test)
        self.rmse = compute_rmse(self.y_pred, self.y_test)
        print('la rmse est :',self.rmse)


        self.mlflow_log_param('name_estimator', self.estimator.__class__.__name__)
        self.mlflow_log_metric('RMSE', self.rmse)

        for key,value in self.dict_params.items():
            self.mlflow_log_metric(key, value)



        return self.rmse

    @memoized_property
    def mlflow_client(self):
        # mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self,filename:str):
        dump(self.pipeline, filename+'.joblib')

print('Class init')





if __name__ == "__main__":
    print('lala')

    trainer3 = Trainer_V3(experiment_name = 'test_experiment')

    trainer3.set_train_test()
    trainer3.set_pipeline(estimator = LinearRegression())
    trainer3.run()
    trainer3.evaluate()
    trainer3.save_model('testsavemodel01')

print('fin execution')


    # trainer2 = Trainer_V2(experiment_name = 'test_experiment')
    # trainer2.set_train_test()
    # trainer2.set_pipeline()
    # trainer2.run()
    # trainer2.evaluate()
    # trainer2.save_model('testsavemodel01')


    # trainer1 = Trainer_V1()
    # df = trainer1.df
    # X = df.drop("fare_amount", axis=1)
    # y = df["fare_amount"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # pipe1 = trainer1.set_pipeline()
    # trained_pipe = trainer1.run(X_train,y_train, pipe1)
    # rmse  = trainer1.evaluate(X_test,y_test,trained_pipe)
    # print('RMSE = ', rmse)
