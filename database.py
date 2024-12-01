from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from typing import Generator
import os

class Database:
    def __init__(self):
        self.DATABASE_URL = os.getenv("DATABASE_URL", "mysql+mysqlconnector://root:password123@localhost/policy-helper")
        self.engine = create_engine(self.DATABASE_URL, echo=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.Base = declarative_base()

    def get_db(self) -> Generator[Session, None, None]:
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()

database = Database()
Base = database.Base
get_db = database.get_db
