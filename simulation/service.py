import pandas as pd
import logging
from .data_model_loader import data_model_loader
from .predictor import Predictor
from .insight_generator import InsightGenerator
from .input_data import InputData
import numpy as np

class SimulationService:
    def __init__(self):
        self.predictor = Predictor()
        self.insight_generator = InsightGenerator()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)  # 로그 레벨을 INFO로 설정
        # 콘솔 핸들러 추가
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # 포맷터 설정
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        # 핸들러를 로거에 추가
        self.logger.addHandler(ch)

    def predict_target_variable(self, input_data: InputData):
        target_variable_name = input_data.target_variable_name
        policy_variable_name = input_data.policy_variable_name
        policy_value = input_data.policy_value
        prediction_length = input_data.prediction_years

        self.logger.info(f"입력 받은 목적변수명: {target_variable_name}")
        self.logger.info(f"입력 받은 정책변수명: {policy_variable_name}")
        self.logger.info(f"입력 받은 정책변수값: {policy_value}")
        self.logger.info(f"입력 받은 예측 기간(year): {prediction_length}")

        df_input = data_model_loader.df_input  # 스케일링된 데이터
        scaler = data_model_loader.scaler  # 스케일러 가져오기
        feature_indices = data_model_loader.feature_indices  # 피처 인덱스 가져오기

        # 변수 존재 여부 확인
        if target_variable_name not in df_input.columns:
            self.logger.info(f"목적변수 '{target_variable_name}'이 데이터에 존재하지 않습니다.")
            return {"error": f"Target variable '{target_variable_name}' not found in data."}

        if policy_variable_name not in df_input.columns:
            self.logger.info(f"정책변수 '{policy_variable_name}'이 데이터에 존재하지 않습니다.")
            return {"error": f"Policy variable '{policy_variable_name}' not found in data."}

        # 외생 변수 식별
        exogenous_variable_names = df_input.columns.difference([target_variable_name, policy_variable_name]).tolist()

        # 데이터 준비
        target = df_input[target_variable_name].values
        # 정책변수와 외생변수를 포함한 피처
        feat_dynamic_real = df_input[[policy_variable_name] + exogenous_variable_names].values.T.tolist()

        # 정책 변수 값 수동 스케일링
        policy_index = feature_indices[policy_variable_name]
        policy_mean = scaler.mean_[policy_index]
        policy_scale = scaler.scale_[policy_index]
        policy_value_scaled = (policy_value - policy_mean) / policy_scale
        future_policy_values = [policy_value_scaled] * prediction_length
        self.logger.info(f"미래 정책변수 값: {future_policy_values}")
        # 미래 외생 변수 값 가져오기 (이미 스케일링됨)
        future_exogenous_values = self._get_future_exogenous_values(exogenous_variable_names, prediction_length)
        self.logger.info(f"미래 외생변수 값: {future_exogenous_values}")
        # 피처 확장
        extended_feat_dynamic_real = [
            feat + future_feat
            for feat, future_feat in zip(feat_dynamic_real, [future_policy_values] + future_exogenous_values)
        ]

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
            prediction_length,
            scaler,
            feature_indices
        )

        # 입력 변수들 정리
        input_variables = {
            "target_variable_name": target_variable_name,
            "policy_variable_name": policy_variable_name,
            "policy_value": policy_value,
            "prediction_length": prediction_length,
            "future_feat_values": future_policy_values,
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

    def _get_future_exogenous_values(self, exogenous_variable_names, prediction_length):
        future_values = []
        for var in exogenous_variable_names:
            # 마지막 스케일링된 값을 사용
            last_value = data_model_loader.df_input[var].dropna().iloc[-1]
            future_values.append([last_value] * prediction_length)
        return future_values

    def _prepare_results(self, df_input, target_variable_name, forecast, prediction_length, scaler, feature_indices):
        forecast_dates = pd.date_range(
            start=df_input.index[-1] + pd.DateOffset(years=1),
            periods=prediction_length,
            freq='Y'
        )

        # 예측된 값 가져오기
        forecast_mean = np.array(forecast.mean.tolist())
        forecast_quantile_30 = np.array(forecast.quantile(0.3).tolist())
        forecast_quantile_70 = np.array(forecast.quantile(0.7).tolist())

        # 목표 변수의 평균과 표준편차 가져오기
        target_index = feature_indices[target_variable_name]
        target_mean = scaler.mean_[target_index]
        target_scale = scaler.scale_[target_index]

        # 예측 결과 역변환
        forecast_mean = forecast_mean * target_scale + target_mean
        forecast_quantile_30 = forecast_quantile_30 * target_scale + target_mean
        forecast_quantile_70 = forecast_quantile_70 * target_scale + target_mean

        predictions = []
        for date, mean, q30, q70 in zip(forecast_dates, forecast_mean, forecast_quantile_30, forecast_quantile_70):
            predictions.append({
                "date": date.strftime('%Y'),
                "mean": mean,
                "quantile_30": q30,
                "quantile_70": q70
            })

        # 히스토리컬 데이터의 목적변수 역변환
        historical_data = df_input[target_variable_name].reset_index().rename(
            columns={'연도': 'date', target_variable_name: 'value'})
        historical_data['value'] = historical_data['value'] * target_scale + target_mean
        historical_data['date'] = historical_data['date'].dt.strftime('%Y')
        historical = historical_data.to_dict(orient='records')


        return predictions, historical
