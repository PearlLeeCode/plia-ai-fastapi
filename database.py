from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from typing import Generator

DATABASE_URL = "mysql+mysqlconnector://root:password123@localhost/policy-helper"

engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 세션 생성

Base = declarative_base()

# 의존성 주입을 위한 세션 생성 함수
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()