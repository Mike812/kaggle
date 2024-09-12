import pickle

from fastapi import FastAPI
from sqlmodel import Field, Session, SQLModel, create_engine, select
from xgboost import XGBClassifier
import pandas as pd

from config import settings
from mental_health.mental_health.mental_health_preprocessing import MentalHealthPreprocessing


class MentalHealthDbEntry(SQLModel, table=True):
    id: int = Field(default=1, primary_key=True)
    statement: str = Field(default=None)
    status: str = Field(index=True)


class UserStatement(SQLModel, table=True):
    statement: str = Field(default=None, primary_key=True)


engine = create_engine(str(settings.SQLALCHEMY_DATABASE_URI))


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


fast_api_app = FastAPI()
# load trained xgboost model
xgb_model: XGBClassifier = pickle.load(open("xgb_mental_health.pkl", "rb"))
train_val_columns = pd.read_csv("train_val_columns.csv")


@fast_api_app.on_event("startup")
def on_startup():
    create_db_and_tables()


@fast_api_app.get("/")
def hello():
    return ("Welcome to my mental health app! Please write a statement with personal feelings and you will get a "
            "prediction to your mental health status.")


@fast_api_app.post("/statement/")
def get_mental_health_status_based_on_statement(statement: UserStatement):
    df_statement = pd.DataFrame(data=[(statement.statement, None)], columns=["statement", "status"])
    # Start preprocessing of test data
    x_test, _ = MentalHealthPreprocessing(df=df_statement, train_val_columns=train_val_columns, col_sum_threshold=0).start()
    status = xgb_model.predict(x_test)
    if status:
        mental_health_status = status[0]
    else:
        mental_health_status = "not defined"
    with Session(engine) as session:
        last_id = get_last_id_from_db(session=session)
        new_db_entry = MentalHealthDbEntry(id=last_id + 1, statement=statement.statement, status=mental_health_status)
        session.add(new_db_entry)
        session.commit()
        session.refresh(new_db_entry)
        return mental_health_status


@fast_api_app.get("/statements/")
def read_db_entries():
    with Session(engine) as session:
        entries = session.exec(select(MentalHealthDbEntry)).all()
        return entries


def get_last_id_from_db(session):
    last_id = session.query(MentalHealthDbEntry).order_by(MentalHealthDbEntry.id.desc()).first()
    if not last_id:
        last_id = 1
    return last_id
