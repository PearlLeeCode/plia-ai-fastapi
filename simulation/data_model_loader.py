import pandas as pd
from .constants import INPUT_DATA_PATH, EXOGENOUS_VARIABLE_PATH  # Add EXOGENOUS_VARIABLE_PATH

class DataModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataModelLoader, cls).__new__(cls)
            cls._instance.df_input = None
            cls._instance.model_dict = {}
        return cls._instance

    def load(self):
        # Load the main input data
        self.df_input = pd.read_csv(INPUT_DATA_PATH)
        self.df_input['연도'] = pd.to_datetime(self.df_input['연도'], format='%Y')
        self.df_input = self.df_input.set_index('연도')

        # Load exogenous variables
        df_exog = pd.read_csv(EXOGENOUS_VARIABLE_PATH)
        df_exog['연도'] = pd.to_datetime(df_exog['연도'], format='%Y')
        df_exog = df_exog.set_index('연도')

        # Merge the exogenous variables with the main dataset
        self.df_input = self.df_input.join(df_exog, how='left')

        self.model_dict = {}

data_model_loader = DataModelLoader()

def load_model_and_data():
    data_model_loader.load()
