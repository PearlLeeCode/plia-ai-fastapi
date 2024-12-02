from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from simulation.router import simulation_router
from simulation.data_model_loader import DataModelLoader
from simulation.predictor import Predictor
from simulation.service import SimulationService

def create_app() -> FastAPI:
    app = FastAPI(
        title="policy-helper",
        description="정책변수에 따른 목적변수 시계열 예측",
        version="0.0.1",
    )

    # CORS 설정
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # http://localhost:5173
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 모델과 데이터 로드
    data_model_loader = DataModelLoader()
    data_model_loader.load()

    # 서비스와 라우터 설정
    predictor = Predictor()
    simulation_service = SimulationService(data_model_loader, predictor)
    app.include_router(simulation_router(simulation_service))

    return app

app = create_app()
