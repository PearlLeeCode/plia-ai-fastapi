from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

app = FastAPI(
    title="policy-helper",
    description="정책변수에 따른 목적변수 시계열 예측",
    version="0.0.1",
)

# Allow CORS for http://localhost:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # The front-end's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class InputData(BaseModel):
    future_childbirth_grant: float
    prediction_years: int = 5  # Default to 5 years if not specified


@app.on_event("startup")
def load_model_and_data():
    global df_input, target, feat_dynamic_real, model_dict
    input_path = "merged_data.csv"
    df_input = pd.read_csv(input_path)

    # Convert '연도' to datetime and set as index
    df_input['연도'] = pd.to_datetime(df_input['연도'], format='%Y')
    df_input = df_input.set_index('연도')

    # Prepare target variable
    target = df_input['합계출산율'].values
    feat_dynamic_real = df_input['첫째아이 평균 출산장려금(천원)'].values.tolist()

    # Initialize an empty dictionary to store models based on prediction length
    model_dict = {}


def get_predictor(prediction_length):
    if prediction_length not in model_dict:
        CTX = len(target)
        model = MoiraiForecast(
            module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
            prediction_length=prediction_length,
            context_length=CTX,
            patch_size="auto",
            num_samples=100,
            target_dim=1,
            feat_dynamic_real_dim=1,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=1)
        model_dict[prediction_length] = predictor
    else:
        predictor = model_dict[prediction_length]
    return predictor


@app.post("/predict", tags=["시뮬레이션"], summary="합계 출산율 예측", description="출산장려금에 따른 합계출산율 예측")
def predict_total_fertility_rate(input_data: InputData):
    future_grant = input_data.future_childbirth_grant * 10  # 만원 단위를 천원 단위로 변환
    prediction_length = input_data.prediction_years

    # Get or create the predictor for the requested prediction length
    predictor = get_predictor(prediction_length)

    # Set future birth incentive values
    future_feat_values = [future_grant] * prediction_length

    # Extend feat_dynamic_real
    extended_feat_dynamic_real = feat_dynamic_real + future_feat_values
    feat_dynamic_real_extended = [extended_feat_dynamic_real]

    # Create prediction dataset
    prediction_data = ListDataset(
        [{
            'start': df_input.index[0],
            'target': target,
            'feat_dynamic_real': feat_dynamic_real_extended
        }],
        freq='Y'
    )

    # Perform prediction
    forecasts = list(predictor.predict(prediction_data))
    forecast = forecasts[0]

    # Prepare results
    forecast_dates = pd.date_range(start=df_input.index[-1] + pd.DateOffset(years=1), periods=prediction_length,
                                   freq='Y')
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

    # Include historical total fertility rate data
    historical_data = df_input['합계출산율'].reset_index().rename(columns={'연도': 'date', '합계출산율': 'total_fertility_rate'})
    historical_data['date'] = historical_data['date'].dt.strftime('%Y')
    historical = historical_data.to_dict(orient='records')

    return {
        "predictions": predictions,
        "historical_data": historical
    }
