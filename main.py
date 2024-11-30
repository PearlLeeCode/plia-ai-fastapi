from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from simulation.router import router as simulation_router
from simulation.dependencies import load_model_and_data

app = FastAPI(
    title="policy-helper",
    description="정책변수에 따른 목적변수 시계열 예측",
    version="0.0.1",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 포함
app.include_router(simulation_router)

# 모델과 데이터 로드
@app.on_event("startup")
def startup_event():
    load_model_and_data()
