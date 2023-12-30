"""
Программа: Модель для построения маски по изображению дорог
Версия: 1.0
"""

import warnings

import uvicorn
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile

from src.evaluate.evaluate import pipeline_evaluate

warnings.filterwarnings("ignore")

app = FastAPI()
CONFIG_PATH = "../config/params.yml"


@app.post("/predict")
def prediction(file: UploadFile = File(...)):
    """
    Предсказание модели по данным из файла
    """
    pipeline_evaluate(config_path=CONFIG_PATH, data_path=file.file)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=80)
