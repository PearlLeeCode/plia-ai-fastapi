import pandas as pd
import logging
from .insight_generator import InsightGenerator
from .input_data import InputData
import numpy as np
from sklearn.model_selection import train_test_split

class SimulationService:
    def __init__(self, data_model_loader, predictor):
        self.data_model_loader = data_model_loader
        self.predictor = predictor
        self.insight_generator = InsightGenerator()
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    def predict_target_variable(self, input_data: InputData):
        self.logger.info(f"입력 데이터: {input_data}")
        df_input = self.data_model_loader.df_input
        scaler = self.data_model_loader.scaler
        feature_indices = self.data_model_loader.feature_indices

        # 데이터 유효성 검사
        validation_result = self._validate_input(df_input, input_data)
        if validation_result is not None:
            return validation_result

        # 예측을 위한 데이터 준비
        target, feat_dynamic_real_extended = self._prepare_prediction_data(df_input, scaler, feature_indices, input_data)

        # 예측 수행
        forecast = self.predictor.predict(
            target,
            feat_dynamic_real_extended,
            df_input.index,
            input_data.prediction_years
        )

        # 결과 준비
        predictions, historical = self._prepare_results(
            df_input,
            input_data.target_variable_name,
            forecast,
            input_data.prediction_years,
            scaler,
            feature_indices
        )

        # 모델 평가
        evaluation_result = self.evaluate_model(input_data.target_variable_name)
        if "error" in evaluation_result:
            return evaluation_result

        mape = evaluation_result["mape"]
        self.logger.info(f"최종 MAPE 값: {mape:.2f}%")

        # 시사점 생성
        insights = self.insight_generator.generate_insights(
            input_data.dict(),
            predictions,
            historical,
            mape
        )

        self.logger.info(f"생성된 시사점 및 고찰: {insights}")

        return {
            "predictions": predictions,
            "historical_data": historical,
            "insights": insights
        }

    def _validate_input(self, df_input, input_data):
        if input_data.target_variable_name not in df_input.columns:
            self.logger.info(f"목적변수 '{input_data.target_variable_name}'이 데이터에 존재하지 않습니다.")
            return {"error": f"Target variable '{input_data.target_variable_name}' not found in data."}

        if input_data.policy_variable_name not in df_input.columns:
            self.logger.info(f"정책변수 '{input_data.policy_variable_name}'이 데이터에 존재하지 않습니다.")
            return {"error": f"Policy variable '{input_data.policy_variable_name}' not found in data."}
        return None

    def _prepare_prediction_data(self, df_input, scaler, feature_indices, input_data):
        target_variable_name = input_data.target_variable_name
        policy_variable_name = input_data.policy_variable_name
        policy_value = input_data.policy_value
        prediction_length = input_data.prediction_years

        exogenous_variable_names = df_input.columns.difference([target_variable_name, policy_variable_name]).tolist()

        target = df_input[target_variable_name].values
        feat_dynamic_real = df_input[[policy_variable_name] + exogenous_variable_names].values.T.tolist()

        policy_index = feature_indices[policy_variable_name]
        policy_mean = scaler.mean_[policy_index]
        policy_scale = scaler.scale_[policy_index]
        policy_value_scaled = (policy_value - policy_mean) / policy_scale
        future_policy_values = [policy_value_scaled] * prediction_length
        self.logger.info(f"미래 정책변수 값: {future_policy_values}")

        future_exogenous_values = self._get_future_exogenous_values(exogenous_variable_names, prediction_length)
        self.logger.info(f"미래 외생변수 값: {future_exogenous_values}")

        extended_feat_dynamic_real = [
            feat + future_feat
            for feat, future_feat in zip(feat_dynamic_real, [future_policy_values] + future_exogenous_values)
        ]

        return target, extended_feat_dynamic_real

    def evaluate_model(self, target_variable_name, test_size=0.2, random_state=42):
        df_input = self.data_model_loader.df_input
        scaler = self.data_model_loader.scaler

        if target_variable_name not in df_input.columns:
            self.logger.error(f"Target variable '{target_variable_name}' not found in data.")
            return {"error": f"Target variable '{target_variable_name}' not found in data."}

        X = df_input.drop(columns=[target_variable_name])
        y = df_input[target_variable_name]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        self.logger.info("X_train shape: " + str(X_train.shape))
        self.logger.info("X_test shape: " + str(X_test.shape))
        self.logger.info("y_train shape: " + str(y_train.shape))
        self.logger.info("y_test shape: " + str(y_test.shape))

        self.logger.info("Training and predicting for evaluation...")

        # 오류 수정: X_train.T.tolist() 대신 X_train.values.T.tolist() 사용
        predictions = self.predictor.predict(y_train.values, X_train.values.T.tolist(), X_test.index, len(X_test))

        target_index = self.data_model_loader.feature_indices[target_variable_name]
        target_mean = scaler.mean_[target_index]
        target_scale = scaler.scale_[target_index]

        y_test_original = y_test * target_scale + target_mean
        predictions_original = predictions.mean * target_scale + target_mean

        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
        self.logger.info(f"Evaluation MAPE: {mape:.2f}%")

        return {"mape": mape}

    def _get_future_exogenous_values(self, exogenous_variable_names, prediction_length):
        future_values = []
        for var in exogenous_variable_names:
            last_value = self.data_model_loader.df_input[var].dropna().iloc[-1]
            future_values.append([last_value] * prediction_length)
        return future_values

    def _prepare_results(self, df_input, target_variable_name, forecast, prediction_length, scaler, feature_indices):
        forecast_dates = pd.date_range(
            start=df_input.index[-1] + pd.DateOffset(years=1),
            periods=prediction_length,
            freq='Y'
        )

        forecast_mean = np.array(forecast.mean.tolist())
        forecast_quantile_30 = np.array(forecast.quantile(0.3).tolist())
        forecast_quantile_70 = np.array(forecast.quantile(0.7).tolist())

        target_index = feature_indices[target_variable_name]
        target_mean = scaler.mean_[target_index]
        target_scale = scaler.scale_[target_index]

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

        historical_data = df_input[target_variable_name].reset_index().rename(
            columns={'연도': 'date', target_variable_name: 'value'})
        target_index = feature_indices[target_variable_name]
        target_mean = scaler.mean_[target_index]
        target_scale = scaler.scale_[target_index]
        historical_data['value'] = historical_data['value'] * target_scale + target_mean
        historical_data['date'] = historical_data['date'].dt.strftime('%Y')
        historical = historical_data.to_dict(orient='records')

        return predictions, historical
