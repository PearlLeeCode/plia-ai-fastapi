import os
import time
import logging
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import openai

class InsightGenerator:
    def __init__(self):
        load_dotenv()
        self.prompt_template = self._create_prompt_template()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.chat_model = ChatOpenAI(
            openai_api_key=self.openai_api_key,
            model_name="gpt-4o-mini-2024-07-18",
            temperature=0,
            max_tokens=2048
        )
        self.logger = logging.getLogger(__name__)

    def _create_prompt_template(self):
        return PromptTemplate(
            input_variables=["input_data", "predictions", "historical", "mape"],
            template="""
당신은 정책 시뮬레이션 결과를 분석하는 정치학자입니다.
정책 시뮬레이션 결과를 바탕으로 시사점 및 고찰을 작성해 주세요. 답변을 통해 입법조사관이 제시하는 정책변수의 값이 목적변수에 미치는 영향을 설명해 주세요. 답변을 통해 입법조사관이 정책 결정에 도움이 되도록 해주세요. 제시하는 정책변수의 값이 목적변수에 미치는 영향에 대해서 창의적으로 해석하고 심도 있게 분석해 주세요. 이를 바탕으로 정책의 효과성을 검토하고자 합니다. 또한 MAPE 값에 대한 분석도 포함해 주세요.
다음은 정책 시뮬레이션에 대한 데이터 및 결과입니다:

입력 데이터:
{input_data}

예측된 목적변수 값:
{predictions}

MAPE: {mape:.2f}%

과거 목적변수 데이터:
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

    def generate_insights(self, input_data, predictions, historical, mape):
        prompt = self.prompt_template.format(
            input_data=input_data,
            predictions=predictions,
            historical=historical,
            mape=mape
        )

        try:
            response = self.chat_model.invoke(prompt)
        except openai.error.RateLimitError as e:
            self.logger.error("Rate limit exceeded. Retrying after a delay...")
            time.sleep(5)
            response = self.chat_model.invoke(prompt)
        except openai.error.InsufficientQuotaError as e:
            self.logger.error("Insufficient quota. Please check your OpenAI account.")
            return "시사점을 생성할 수 없습니다. OpenAI API의 사용 한도를 초과하였습니다."

        insights = response.content.strip()
        return insights
