import pandas as pd
from .constants import INPUT_DATA_PATH, EXOGENOUS_VARIABLE_PATH
from sklearn.preprocessing import StandardScaler

class DataModelLoader:
    def __init__(self):
        self.df_input = None
        self.scaler = None
        self.feature_indices = None

    def load(self):
        self._load_data()
        self._merge_exogenous_variables()
        self._scale_data()
        self._create_feature_indices()

    def _load_data(self):
        self.df_input = pd.read_csv(INPUT_DATA_PATH)
        self.df_input['연도'] = pd.to_datetime(self.df_input['연도'], format='%Y')
        self.df_input = self.df_input.set_index('연도')

    def _merge_exogenous_variables(self):
        df_exog = pd.read_csv(EXOGENOUS_VARIABLE_PATH)
        df_exog['연도'] = pd.to_datetime(df_exog['연도'], format='%Y')
        df_exog = df_exog.set_index('연도')
        self.df_input = self.df_input.join(df_exog, how='left')

    def _scale_data(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.df_input)
        self.df_input[self.df_input.columns] = self.scaler.transform(self.df_input)

    def _create_feature_indices(self):
        self.feature_indices = {feature: idx for idx, feature in enumerate(self.df_input.columns)}
