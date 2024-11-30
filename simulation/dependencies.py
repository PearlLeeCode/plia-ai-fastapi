import pandas as pd
from .constants import INPUT_DATA_PATH

class DataModelLoader:
    def __init__(self):
        self.df_input = None
        self.model_dict = {}

    def load(self):
        self.df_input = pd.read_csv(INPUT_DATA_PATH)
        self.df_input['연도'] = pd.to_datetime(self.df_input['연도'], format='%Y')
        self.df_input = self.df_input.set_index('연도')
        self.model_dict = {}

data_model_loader = DataModelLoader()

def load_model_and_data():
    data_model_loader.load()
