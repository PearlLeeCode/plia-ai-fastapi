from fastapi import APIRouter
from .schemas import InputData
from .service import SimulationService

router = APIRouter(
    prefix="/simulation",
    tags=["시뮬레이션"]
)

simulation_service = SimulationService()

@router.post("/predict", summary="정책변수값 따른 미래 목적변수 예측", description="목적변수명, 예측할 목적변수의 기간(year), 정책변수명, 제시할 정책변수값을 입력하면 목적변수의 과거값과 예측값을 함께 반환한다.")
def predict(input_data: InputData):
    return simulation_service.predict_target_variable(input_data)
