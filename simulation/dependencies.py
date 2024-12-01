import pandas as pd
from .constants import INPUT_DATA_PATH

class DataModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataModelLoader, cls).__new__(cls)
            cls._instance.df_input = None
            cls._instance.model_dict = {}
        return cls._instance

    def load(self):
        self.df_input = pd.read_csv(INPUT_DATA_PATH)
        self.df_input['연도'] = pd.to_datetime(self.df_input['연도'], format='%Y')
        self.df_input = self.df_input.set_index('연도')
        self.model_dict = {}

data_model_loader = DataModelLoader()

def load_model_and_data():
    data_model_loader.load()
