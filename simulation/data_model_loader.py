import pandas as pd
from .constants import INPUT_DATA_PATH, EXOGENOUS_VARIABLE_PATH
from sklearn.preprocessing import StandardScaler

class DataModelLoader:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataModelLoader, cls).__new__(cls)
            cls._instance.df_input = None
            cls._instance.model_dict = {}
            cls._instance.scaler = None
            cls._instance.feature_indices = None  # 추가
        return cls._instance

    def load(self):
        # 메인 데이터 로드
        self.df_input = pd.read_csv(INPUT_DATA_PATH)
        self.df_input['연도'] = pd.to_datetime(self.df_input['연도'], format='%Y')
        self.df_input = self.df_input.set_index('연도')

        # 외생 변수 로드
        df_exog = pd.read_csv(EXOGENOUS_VARIABLE_PATH)
        df_exog['연도'] = pd.to_datetime(df_exog['연도'], format='%Y')
        df_exog = df_exog.set_index('연도')

        # 데이터 병합
        self.df_input = self.df_input.join(df_exog, how='left')

        # 데이터 스케일링
        self.scaler = StandardScaler()
        self.scaler.fit(self.df_input)
        self.df_input[self.df_input.columns] = self.scaler.transform(self.df_input)

        # 피처 인덱스 저장
        self.feature_indices = {feature: idx for idx, feature in enumerate(self.df_input.columns)}

        self.model_dict = {}

data_model_loader = DataModelLoader()

def load_model_and_data():
    data_model_loader.load()
