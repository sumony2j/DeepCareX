FROM python:3.10

RUN apt update -y

RUN apt upgrade -y

RUN pip install --upgrade pip

RUN pip install numpy pandas scikit-learn matplotlib scipy seaborn xgboost joblib flask tensorflow==2.9 db-sqlite3

ENV TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /DeepCareX

ADD Website Website/

ADD Models Models/

EXPOSE 5000

WORKDIR /DeepCareX/Website/database/

RUN python3 database.py

WORKDIR /DeepCareX/Website

ENV  FLASK_APP main.py

ENV FLASK_DEBUG 1

CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0"]

