import pandas as pd
import sklearn.model_selection as msel
import typing
import numpy as np


class DataFromCSV:

    __instance__ = None

    def __init__(self, filename: typing.Union[None, str] = None):

        self.csv_fname: typing.Union[None, str] = None
        self.data: typing.Union[None, pd.DataFrame] = None
        self.process_data: typing.Union[None, pd.DataFrame] = None
        self.list_str_data: typing.List[str] = []
        self.list_num_data: typing.List[str] = []
        self.feature_list: typing.List[str] = []
        self.predict_target: typing.Union[None, str] = None

        if filename is not None:
            self.get_data(filename)
        # self.train_X: typing.Union[None, pd.DataFrame] = None
        # self.train_y: typing.Union[None, pd.DataFrame] = None
        # self.val_X: typing.Union[None, pd.DataFrame] = None
        # self.val_y: typing.Union[None, pd.DataFrame] = None
        DataFromCSV.__instance__ = self

    def __del__(self):
        DataFromCSV.__instance__ = None

    def get_data(self, filename: str) -> pd.DataFrame:

        self.csv_fname = filename
        data = pd.read_csv(filename)
        self.data = data
        print(self.data.columns)
        return data

    def create_process_data(self):
        self.process_data = self.data.copy()

    # convert categorical variables to dummy 0/1 variables
    def str_to_categorize_code(self, col: typing.List[str]):

        encode_data = pd.get_dummies(self.data, columns=col)
        print(' === Converting Categorical variables ===')
        print(col)
        print(' ------------------------------ ')
        print(encode_data.columns)
        print(' ------------------------------ ')
        print(len(encode_data.columns))
        print(' ------------------------------ ')
        print(encode_data.head())
        if self.process_data is None:
            self.data = encode_data
        else:
            self.process_data = encode_data

    # Transfer string categorized data to numerical code. str_code: { str_data: numerical_code }
    # Add nan option for missing data or data = nan
    def categorize_new_feature(self, col_name: str, str_code: typing.Dict, add_nan_opt: bool = False):

        n_val = len(self.data[col_name].unique())
        n_code = len(str_code)

        col_name_other = col_name + '_other'
        col_name_nan = col_name + '_NA'
        if n_code < n_val:
            self.data[col_name_other] = self.data.apply(lambda x: 1, axis=1)
        if add_nan_opt:
            self.data[col_name_nan] = self.data.apply(
                lambda x: 1 if x[col_name] is np.nan else 0, axis=1
            )
            if n_code < n_val:
                self.data[col_name_other] = self.data.apply(
                    lambda x: 0 if x[col_name] is np.nan else x[col_name_other], axis=1
                )

        new_feature_list = []
        for key, val in str_code.items():
            col_name_key = col_name + '_' + key
            new_feature_list.append(col_name_key)
            self.data[col_name_key] = self.data.apply(lambda x: val if x[col_name] == key else 0, axis=1)
            if n_code < n_val:
                self.data[col_name_other] = self.data.apply(
                    lambda x: 0 if x[col_name] == key else x[col_name_other], axis=1
                )

        '''
        # verification
        for j in range(100):
            i = j +33
            print(' %s = %d , %d, %d' % (
                self.data[col_name].iloc[i],
                self.data[new_feature_list[0]].iloc[i],
                self.data[new_feature_list[1]].iloc[i],
                # self.data[col_name_nan].iloc[i],
                self.data[col_name_other].iloc[i]
            ))
        '''
    # Digitalized str variables to numbers
    def digi_code_str_feature(self, col_name: str):

        if self.process_data is None:
            self.process_data = self.data

        n_val = len(self.process_data[col_name].unique())

        # Assign number to each string feature value
        code_dict = {}
        code_it = 0
        for it in self.process_data[col_name].unique():
            code_dict[it] = code_it
            # print(' feature[%s] = %d ' % (it, code_it))
            code_it = code_it + 1

        # verification
        # for j in range(10):
        #     i = j + 15
        #     print(' Origin : %d = %s' % (i, process_data[col_name].iloc[i]))

        # Renew the value
        for key, val in code_dict.items():
            self.process_data[col_name] = self.process_data.apply(
                lambda x: val if x[col_name] == key else x[col_name],
                axis=1
            )

        # verification
        # for j in range(10):
        #     i = j +15
        #     print(' New Code: %d = %d ' % (i, process_data[col_name].iloc[i]))

    def combined_num_data_plus(self, column_a: str, column_b: str, new_column: str):
        if self.process_data is None:
            self.process_data = self.data
        self.process_data[new_column] = self.process_data.apply(lambda x: x[column_a] + x[column_b], axis=1)

    def combined_num_data_minus(self, column_a: str, column_b: str, new_column: str):
        if self.process_data is None:
            self.process_data = self.data
        self.process_data[new_column] = self.process_data.apply(lambda x: x[column_a] - x[column_b], axis=1)

    def combined_num_data_and(self, column_a: str, column_b: str, new_column: str):
        if self.process_data is None:
            self.process_data = self.data
        self.process_data[new_column] = self.process_data.apply(lambda x: x[column_a] and x[column_b], axis=1)

    def combined_num_data_or(self, column_a: str, column_b: str, new_column: str):
        if self.process_data is None:
            self.process_data = self.data
        self.process_data[new_column] = self.process_data.apply(lambda x: x[column_a] or x[column_b], axis=1)

    def identify_num_str_data(self):

        self.list_str_data = []
        self.list_num_data = []
        for it in self.data.columns:
            if self.data[it].dtypes == object:
                self.list_str_data.append(it)
            else:
                self.list_num_data.append(it)

        print(' len of all data = %d ' % len(self.data.columns))
        print(' len of str data = %d ' % len(self.list_str_data))
        for it in self.list_str_data:
            print(' --- Name: %s (%d)--- ' % (it, len(self.data[it].unique())))
            print(self.data[it].unique())
            print(' ------ value counts --------- ')
            print(self.data[it].value_counts())
            print(' ------------------------------- ')
        for it in self.list_num_data:
            print(' -- Numerical Data (%s) type : %s ' % (it, self.data[it].dtypes))

        print(self.list_str_data)
        print(self.list_num_data)

    def set_predict_target(self, target_name: str):
        self.predict_target = target_name

    def set_feature_list(self, features: typing.List[str]):

        for it in features:
            if it in self.feature_list:
                print(' Feature %s already exist in the list !' % it)
            else:
                self.feature_list.append(it)

        print(' Current Feature List')
        print(self.feature_list)

    def feature_selection(self, col_name: str, feature_list: typing.List):

        sel_data = self.data[self.data[col_name].isin(feature_list)]
        self.data = sel_data

    def create_feature_data(self,
                            feature_list: typing.Union[None, typing.List[str]],
                            predict_target: typing.Union[None, str],
                            test_size: float = 0.3,
                            use_process_data: bool = True):

        if feature_list is None:
            feature_list = self.feature_list
        if predict_target is None:
            predict_target = self.predict_target

        print('== Predict Target is %s' % predict_target)

        if use_process_data is True:
            sel_data_x = self.process_data[feature_list]
            sel_data_y = self.process_data[predict_target]
        else:
            sel_data_x = self.data[feature_list]
            sel_data_y = self.data[predict_target]

        train_X, val_X, train_y, val_y = msel.train_test_split(
            sel_data_x,
            sel_data_y,
            test_size=test_size
        )
        return train_X, val_X, train_y, val_y

    def create_training_testing_data(self, predict_target: str, test_size: float = 0.3):

        # self.str_to_categorize_code(['Heating', 'CentralAir'])
        self.categorize_new_feature('GarageType', {'Attchd': 1, 'Detchd': 1}, add_nan_opt=True)
        self.categorize_new_feature('CentralAir', {'Y': 1}, add_nan_opt=False)
        self.digi_code_str_feature('Neighborhood')

        feature_list = ['YearBuilt', 'LotArea', 'BedroomAbvGr', 'GrLivArea', 'BsmtFinSF1',
                        'FullBath', 'HalfBath', 'Neighborhood',
                        'OverallQual', 'OverallCond',
                        'GarageCars', 'GarageType_Attchd', 'GarageType_Detchd',
                        'CentralAir_Y']

        train_data_x = self.data[feature_list]
        train_data_y = self.data[predict_target]
        train_X, val_X, train_y, val_y = msel.train_test_split(
            train_data_x,
            train_data_y,
            test_size=test_size
        )
        print('=========== Train X =================')
        print(train_data_x)
        return train_X, val_X, train_y, val_y

    @staticmethod
    def get_instance():
        if DataFromCSV.__instance__ is None:
            DataFromCSV.__instance__ = DataFromCSV()
        # end if
        return DataFromCSV.__instance__
    # end get_instance
