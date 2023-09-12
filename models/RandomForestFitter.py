import numpy as np
import pandas as pd
import typing
# import sklearn.metrics as mtx
import data_processor.read_data as read_data
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RandomForestFitter:

    def __init__(self):

        # RandomForest Model parameters
        self.n_tree: int = 100
        self.criterion: str = 'gini'
        self.max_feature: typing.Union[None, int] = None
        self.max_depth: typing.Union[None, int] = None
        self.min_samples_split: typing.Union[int, float] = 2
        self.min_samples_leaf: typing.Union[int, float] = 1
        self.model = None

        # Training and validation data set
        self.data_obj = read_data.DataFromCSV.get_instance()
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.target: str = ''
        # Model performance
        self.percent_err: typing.List[float] = []

    def config_model(self, n_tree: int,
                     model_type: str,
                     criterion: str,
                     max_feature: typing.Union[None, int],
                     max_depth: typing.Union[None, int],
                     min_samples_split: typing.Union[int, float] = 2,
                     min_samples_leaf: typing.Union[int, float] = 1
                     ):

        if model_type == 'Regressor':
            self.model = RandomForestRegressor(n_tree,
                                               criterion=criterion,
                                               max_depth=max_depth,
                                               max_features=max_feature,
                                               min_samples_split=min_samples_split,
                                               min_samples_leaf=min_samples_leaf)
        if model_type == 'Classifier':
            self.model = RandomForestClassifier(n_tree,
                                                criterion=criterion,
                                                max_depth=max_depth,
                                                max_features=max_feature,
                                                min_samples_split=min_samples_split,
                                                min_samples_leaf=min_samples_leaf)

        self.n_tree = n_tree
        self.criterion = criterion
        self.max_feature = max_feature
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def set_predict_target(self, target_name: str):
        self.target = target_name

    def get_training_data(self, file_path: str, test_size: float = 0.3, target_name: str = ''):

        if self.data_obj is None:
            self.data_obj = read_data.DataFromCSV()

        self.data_obj.get_data(file_path)
        if target_name == '':
            target_name = self.target
        self.train_x, self.val_x, self.train_y, self.val_y = self.data_obj.create_training_testing_data(
            target_name, test_size=test_size)

    def input_training_data(self, train_x, val_x, train_y, val_y):
        self.train_x = train_x
        self.val_x = val_x
        self.train_y = train_y
        self.val_y = val_y

    def get_test_data(self, file_path: str):
        if self.data_obj is None:
            self.data_obj = read_data.DataFromCSV()
        test_data = pd.read_csv(file_path)
        print(' Test data # of columns = %d' % len(test_data.columns))
        self.data_obj.create_feature_data()
        print(test_data.columns)

    def fit(self):
        self.model.fit(self.train_x, self.train_y)

    def get_predict(self, validate_x: pd.DataFrame, validate_y: pd.DataFrame) -> np.ndarray:
        predict = self.model.predict(validate_x)
        print(' Predict  : (%d) ' % len(predict))
        for i in range(len(predict)):
            prct_err = (validate_y.iloc[i]-predict[i])/validate_y.iloc[i]
            self.percent_err.append(prct_err)
            print('predict = %.1f , actual = %.1f , diff = %.3f'
                  % (predict[i], validate_y.iloc[i], prct_err))

        return predict
        # error = mtx.mean_squared_error(val_y, predict)
        # print(' Error : ')
        # print(error)
