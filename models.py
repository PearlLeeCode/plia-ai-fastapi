from sqlalchemy import Column, Integer, String, Float, ForeignKey, TIMESTAMP
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

# target_variables 테이블 모델
class TargetVariable(Base):
    __tablename__ = "target_variables"

    target_variable_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)  # 예: "합계출산율"

    # Add this relationship
    simulation_requests = relationship("SimulationRequest", back_populates="target_variable")


# policy_variables 테이블 모델
class PolicyVariable(Base):
    __tablename__ = "policy_variables"

    policy_variable_id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)  # 예: "첫째아이 평균 출산장려금", "육아휴직급여"

    # Add this relationship
    simulation_requests = relationship("SimulationRequest", back_populates="policy_variable")


# simulation_request 테이블 모델
class SimulationRequest(Base):
    __tablename__ = "simulation_request"

    request_id = Column(Integer, primary_key=True, index=True)
    target_variable_id = Column(Integer, ForeignKey('target_variables.target_variable_id'))
    policy_variable_id = Column(Integer, ForeignKey('policy_variables.policy_variable_id'))
    request_forecast_years = Column(Integer)
    proposed_policy_value = Column(Float)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    target_variable = relationship("TargetVariable", back_populates="simulation_requests")
    policy_variable = relationship("PolicyVariable", back_populates="simulation_requests")
    forecast_data = relationship("ForecastData", back_populates="simulation_request")


# forecast_data 테이블 모델
class ForecastData(Base):
    __tablename__ = "forecast_data"

    forecast_data_id = Column(Integer, primary_key=True, index=True)
    request_id = Column(Integer, ForeignKey('simulation_request.request_id'))
    forecast_year = Column(Integer)
    mean = Column(Float)
    quantile_30 = Column(Float)
    quantile_70 = Column(Float)

    simulation_request = relationship("SimulationRequest", back_populates="forecast_data")
