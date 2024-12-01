import pandas as pd
from gluonts.dataset.common import ListDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from .dependencies import data_model_loader
import logging
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import openai
import time

logging.basicConfig(level=logging.INFO)

load_dotenv()  # 환경 변수 로드

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

def generate_insights(input_variables, predictions, historical):
    # LangChain을 사용하여 프롬프트 템플릿 생성
    prompt_template = PromptTemplate(
        input_variables=["target_variable_name", "policy_variable_name", "policy_value", "prediction_length",
                         "future_feat_values", "extended_feat_dynamic_real", "predictions", "historical"],
        template="""
당신은 정책 시뮬레이션 결과를 분석하는 정치학자입니다.
정책 시뮬레이션 결과를 바탕으로 시사점 및 고찰을 작성해 주세요. 답변을 통해 입법조사관이 제시하는 정책변수의 값이 목적변수에 미치는 영향을 설명해 주세요. 답변을 통해 입법조사관이 정책 결정에 도움이 되도록 해주세요. 제시하는 정책변수의 값이 목적변수에 미치는 영향에 대해서 창의적으로 해석하고 심도 있게 분석해 주세요. 이를 바탕으로 정책의 효과성을 검토하고자 합니다.
다음은 정책 시뮬레이션 결과입니다:

입력 받은 목적변수명: {target_variable_name}
입력 받은 정책변수명: {policy_variable_name}
입력 받은 정책변수값: {policy_value}
입력 받은 예측 기간(year): {prediction_length}
미래 정책변수값: {future_feat_values}
확장된 정책변수 데이터: {extended_feat_dynamic_real}

예측된 목적변수 값:
{predictions}

{target_variable_name}의 과거 데이터:
{historical}

답변양식:

정책 시뮬레이션 결과에 대한 분석

1. 목적

2. 현황

3. 분석 결과

4. 시사점 및 고찰

5. 제언

6. 기대효과

7. 결론

답변 작성시 주의사항:
답변의 각 항목에는 * 등의 기호를 사용하지 마세요.
"""
    )

    prompt = prompt_template.format(
        target_variable_name=input_variables['target_variable_name'],
        policy_variable_name=input_variables['policy_variable_name'],
        policy_value=input_variables['policy_value'],
        prediction_length=input_variables['prediction_length'],
        future_feat_values=input_variables['future_feat_values'],
        extended_feat_dynamic_real=input_variables['extended_feat_dynamic_real'],
        predictions=predictions,
        historical=historical
    )

    # OpenAI API 키 설정 (환경 변수 사용)
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # LangChain의 ChatOpenAI를 사용하여 모델 생성
    chat = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name="gpt-4o-mini-2024-07-18",  # 모델명 수정
        temperature=0,
        max_tokens=2048
    )

    # 메시지 생성 및 응답 받기 - 'invoke' 메서드 사용
    try:
        response = chat.invoke(prompt)
    except openai.error.RateLimitError as e:
        logging.error("Rate limit exceeded. Retrying after a delay...")
        time.sleep(5)  # 5초 후 재시도
        response = chat.invoke(prompt)
    except openai.error.InsufficientQuotaError as e:
        logging.error("Insufficient quota. Please check your OpenAI account.")
        return "시사점을 생성할 수 없습니다. OpenAI API의 사용 한도를 초과하였습니다."

    insights = response.content.strip()
    return insights

def predict_target_variable(input_data):
    target_variable_name = input_data.target_variable_name
    logging.info(f"입력 받은 목적변수명: {target_variable_name}")
    policy_variable_name = input_data.policy_variable_name
    logging.info(f"입력 받은 정책변수명: {policy_variable_name}")
    policy_value = input_data.policy_value
    logging.info(f"입력 받은 정책변수값: {policy_value}")
    prediction_length = input_data.prediction_years
    logging.info(f"입력 받은 예측 기간(year): {prediction_length}")

    df_input = data_model_loader.df_input  # 데이터셋 로드

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

    # 입력 변수들 정리
    input_variables = {
        "target_variable_name": target_variable_name,
        "policy_variable_name": policy_variable_name,
        "policy_value": policy_value,
        "prediction_length": prediction_length,
        "future_feat_values": future_feat_values,
        "extended_feat_dynamic_real": extended_feat_dynamic_real
    }

    # 시사점 및 고찰 생성
    insights = generate_insights(input_variables, predictions, historical)
    logging.info(f"생성된 시사점 및 고찰: {insights}")
    return {
        "predictions": predictions,
        "historical_data": historical,
        "insights": insights
    }
