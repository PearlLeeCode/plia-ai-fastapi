import pandas as pd
import logging
from .data_model_loader import data_model_loader
from .predictor import Predictor
from .insight_generator import InsightGenerator
from .input_data import InputData

class SimulationService:
    def __init__(self):
        self.predictor = Predictor()
        self.insight_generator = InsightGenerator()
        self.logger = logging.getLogger(__name__)

        # 로깅 설정 추가
        handler = logging.StreamHandler()  # 콘솔에 출력하기 위한 핸들러
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # 로그 메시지 포맷
        handler.setFormatter(formatter)

        self.logger.addHandler(handler)  # 핸들러 추가
        self.logger.setLevel(logging.INFO)  # 로그 레벨 설정 (INFO 레벨 이상은 출력)

    def predict_target_variable(self, input_data: InputData):
        target_variable_name = input_data.target_variable_name
        policy_variable_name = input_data.policy_variable_name
        policy_value = input_data.policy_value
        prediction_length = input_data.prediction_years

        self.logger.info(f"입력 받은 목적변수명: {target_variable_name}")
        self.logger.info(f"입력 받은 정책변수명: {policy_variable_name}")
        self.logger.info(f"입력 받은 정책변수값: {policy_value}")
        self.logger.info(f"입력 받은 예측 기간(year): {prediction_length}")

        df_input = data_model_loader.df_input  # Load the dataset

        # Check if variables exist
        if target_variable_name not in df_input.columns:
            self.logger.info(f"목적변수 '{target_variable_name}'이 데이터에 존재하지 않습니다.")
            return {"error": f"Target variable '{target_variable_name}' not found in data."}

        if policy_variable_name not in df_input.columns:
            self.logger.info(f"정책변수 '{policy_variable_name}'이 데이터에 존재하지 않습니다.")
            return {"error": f"Policy variable '{policy_variable_name}' not found in data."}

        # Identify exogenous variables (exclude target and policy variables)
        exogenous_variable_names = df_input.columns.difference([target_variable_name, policy_variable_name]).tolist()
        self.logger.info(f"외생변수: {exogenous_variable_names}")

        # Prepare data
        target = df_input[target_variable_name].values
        # Combine policy variable and exogenous variables
        feat_dynamic_real = df_input[[policy_variable_name] + exogenous_variable_names].values.T.tolist()

        # Prepare future values for policy variable
        future_policy_values = [policy_value] * prediction_length
        self.logger.info(f"미래 정책변수값: {future_policy_values}")
        # Get future values for exogenous variables
        future_exogenous_values = self._get_future_exogenous_values(exogenous_variable_names, prediction_length)
        self.logger.info(f"미래 외생변수값: {future_exogenous_values}")
        # Extend features with future values
        extended_feat_dynamic_real = [
            feat + future_feat
            for feat, future_feat in zip(feat_dynamic_real, [future_policy_values] + future_exogenous_values)
        ]
        self.logger.info(f"확장된 정책변수 데이터: {extended_feat_dynamic_real}")
        # Perform prediction
        forecast = self.predictor.predict(
            target,
            extended_feat_dynamic_real,
            df_input.index,
            prediction_length
        )

        # Process results
        predictions, historical = self._prepare_results(
            df_input,
            target_variable_name,
            forecast,
            prediction_length
        )

        # Organize input variables
        input_variables = {
            "target_variable_name": target_variable_name,
            "policy_variable_name": policy_variable_name,
            "policy_value": policy_value,
            "prediction_length": prediction_length,
            "future_feat_values": future_policy_values,
            "extended_feat_dynamic_real": extended_feat_dynamic_real
        }

        # Generate insights
        insights = self.insight_generator.generate_insights(
            input_variables,
            predictions,
            historical
        )

        self.logger.info(f"생성된 시사점 및 고찰: {insights}")
        return {
            "predictions": predictions,
            "historical_data": historical,
            "insights": insights
        }

    def _get_future_exogenous_values(self, exogenous_variable_names, prediction_length):
        future_values = []
        for var in exogenous_variable_names:
            # Use the last observed value for each exogenous variable
            last_value = data_model_loader.df_input[var].dropna().iloc[-1]
            future_values.append([last_value] * prediction_length)
        return future_values

    def _prepare_results(self, df_input, target_variable_name, forecast, prediction_length):
        forecast_dates = pd.date_range(
            start=df_input.index[-1] + pd.DateOffset(years=1),
            periods=prediction_length,
            freq='Y'
        )

        forecast_mean = forecast.mean.tolist()
        forecast_quantile_30 = forecast.quantile(0.3).tolist()
        forecast_quantile_70 = forecast.quantile(0.7).tolist()

        predictions = []
        for date, mean, q30, q70 in zip(forecast_dates, forecast_mean, forecast_quantile_30, forecast_quantile_70):
            predictions.append({
                "date": date.strftime('%Y'),
                "mean": mean,
                "quantile_30": q30,
                "quantile_70": q70
            })

        historical_data = df_input[target_variable_name].reset_index().rename(
            columns={'연도': 'date', target_variable_name: 'value'})
        historical_data['date'] = historical_data['date'].dt.strftime('%Y')
        historical = historical_data.to_dict(orient='records')

        return predictions, historical
