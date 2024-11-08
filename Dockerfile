FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

COPY ./app /app

RUN pip install joblib pandas scikit-learn

COPY ./model/model.pkl /app/model.pkl
