import pandas as pd  # pandas import 추가
import logging
from .dependencies import data_model_loader
from .predictor import Predictor
from .insight_generator import InsightGenerator
from .schemas import InputData

class SimulationService:
    def __init__(self):
        self.predictor = Predictor()
        self.insight_generator = InsightGenerator()
        self.logger = logging.getLogger(__name__)

    def predict_target_variable(self, input_data: InputData):
        target_variable_name = input_data.target_variable_name
        policy_variable_name = input_data.policy_variable_name
        policy_value = input_data.policy_value
        prediction_length = input_data.prediction_years

        self.logger.info(f"입력 받은 목적변수명: {target_variable_name}")
        self.logger.info(f"입력 받은 정책변수명: {policy_variable_name}")
        self.logger.info(f"입력 받은 정책변수값: {policy_value}")
        self.logger.info(f"입력 받은 예측 기간(year): {prediction_length}")

        df_input = data_model_loader.df_input  # 데이터셋 로드

        # 변수 존재 여부 확인
        if target_variable_name not in df_input.columns:
            self.logger.info(f"목적변수 '{target_variable_name}'이 데이터에 존재하지 않습니다.")
            return {"error": f"Target variable '{target_variable_name}' not found in data."}

        if policy_variable_name not in df_input.columns:
            self.logger.info(f"정책변수 '{policy_variable_name}'이 데이터에 존재하지 않습니다.")
            return {"error": f"Policy variable '{policy_variable_name}' not found in data."}

        # 데이터 준비
        target = df_input[target_variable_name].values
        feat_dynamic_real = df_input[policy_variable_name].values.tolist()
        future_feat_values = [policy_value] * prediction_length
        extended_feat_dynamic_real = feat_dynamic_real + future_feat_values

        # 예측 수행
        forecast = self.predictor.predict(
            target,
            extended_feat_dynamic_real,
            df_input.index,
            prediction_length
        )

        # 결과 처리
        predictions, historical = self._prepare_results(
            df_input,
            target_variable_name,
            forecast,
            prediction_length
        )

        # 입력 변수들 정리
        input_variables = {
            "target_variable_name": target_variable_name,
            "policy_variable_name": policy_variable_name,
            "policy_value": policy_value,
            "prediction_length": prediction_length,
            "future_feat_values": future_feat_values,
            "extended_feat_dynamic_real": extended_feat_dynamic_real
        }

        # 시사점 생성
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
