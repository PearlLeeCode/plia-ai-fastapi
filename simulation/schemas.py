from pydantic import BaseModel

class InputData(BaseModel):
    target_variable_name: str  # 목적변수명
    policy_variable_name: str  # 정책변수명
    policy_value: float        # 정책변수값
    prediction_years: int = 5  # 예측 년수 (기본값: 5년)
