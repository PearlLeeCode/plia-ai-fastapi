import pandas as pd
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from .dependencies import data_model_loader
import logging
logging.basicConfig(level=logging.INFO)

def get_predictor(prediction_length, context_length):
    key = (prediction_length, context_length)
    if key not in data_model_loader.model_dict:
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
            prediction_length=prediction_length,
            context_length=context_length,
            patch_size="auto",
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=1,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=1)
        data_model_loader.model_dict[key] = predictor
    else:
        predictor = data_model_loader.model_dict[key]
    return predictor

def predict_target_variable(input_data):
    target_variable_name = input_data.target_variable_name
    logging.info(f"입력 받은 목적변수명: {target_variable_name}")
    policy_variable_name = input_data.policy_variable_name
    logging.info(f"입력 받은 정책변수명: {policy_variable_name}")
    policy_value = input_data.policy_value
    logging.info(f"입력 받은 정책변수값: {policy_value}")
    prediction_length = input_data.prediction_years
    logging.info(f"입력 받은 예측 기간(year): {prediction_length}")

    df_input = data_model_loader.df_input # 데이터셋 로드

    # 변수 존재 여부 확인
    if target_variable_name not in df_input.columns:
        logging.info(f"목적변수 '{target_variable_name}'이 csv 파일에 존재하지 않습니다.")
        return {"error": f"Target variable '{target_variable_name}' not found in data."}

    if policy_variable_name not in df_input.columns:
        logging.info(f"정책변수 '{policy_variable_name}'이 csv 파일에 존재하지 않습니다.")
        return {"error": f"Policy variable '{policy_variable_name}' not found in data."}

    # 목적변수와 정책변수 데이터 추출
    target = df_input[target_variable_name].values
    logging.info(f"목적변수 '{target_variable_name}' 데이터: {target}")
    feat_dynamic_real = df_input[policy_variable_name].values.tolist()
    logging.info(f"정책변수 '{policy_variable_name}' 데이터: {feat_dynamic_real}")

    # 미래 정책변수값 설정
    future_feat_values = [policy_value] * prediction_length
    logging.info(f"미래 정책변수값: {future_feat_values}")

    # feat_dynamic_real 확장
    extended_feat_dynamic_real = feat_dynamic_real + future_feat_values
    logging.info(f"확장된 정책변수 데이터: {extended_feat_dynamic_real}")
    feat_dynamic_real_extended = [extended_feat_dynamic_real]

    # 예측 데이터셋 생성
    prediction_data = ListDataset(
        [{
            'start': df_input.index[0],
            'target': target,
            'feat_dynamic_real': feat_dynamic_real_extended
        }],
        freq='Y'
    )

    # 예측기 가져오기
    context_length = len(target)
    predictor = get_predictor(prediction_length, context_length)

    # 예측 수행
    forecasts = list(predictor.predict(prediction_data))
    forecast = forecasts[0]

    # 결과 준비
    forecast_dates = pd.date_range(
        start=df_input.index[-1] + pd.DateOffset(years=1),
        periods=prediction_length,
        freq='Y'
    )

    forecast_mean = forecast.mean.tolist()
    logging.info(f"예측된 목적변수 평균값: {forecast_mean}")
    forecast_quantile_30 = forecast.quantile(0.3).tolist()
    logging.info(f"예측된 목적변수 30% 분위값: {forecast_quantile_30}")
    forecast_quantile_70 = forecast.quantile(0.7).tolist()
    logging.info(f"예측된 목적변수 70% 분위값: {forecast_quantile_70}")

    predictions = []
    for date, mean, q30, q70 in zip(forecast_dates, forecast_mean, forecast_quantile_30, forecast_quantile_70):
        predictions.append({
            "date": date.strftime('%Y'),
            "mean": mean,
            "quantile_30": q30,
            "quantile_70": q70
        })

    # 과거 목적변수 데이터 포함
    historical_data = df_input[target_variable_name].reset_index().rename(
        columns={'연도': 'date', target_variable_name: 'value'})
    historical_data['date'] = historical_data['date'].dt.strftime('%Y')
    historical = historical_data.to_dict(orient='records')

    return {
        "predictions": predictions,
        "historical_data": historical
    }
