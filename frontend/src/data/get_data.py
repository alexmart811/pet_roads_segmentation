"""
Программа: Получение данных по пути и чтение
Версия: 1.0
"""

from typing import Any, Text, Dict, Tuple
import io
import pickle
from PIL import Image


def from_pickle(path: str) -> Any:
    """
    Загрузка данных по пути с формата pickle
    :params path: путь до файла
    :return: данне любого типа, которые были записаны в файле
    """
    with open(path, "rb") as f:
        file = pickle.load(f)
    return file


def get_image(image_path: Text) -> Dict[str, Tuple[str, io.BytesIO, str]]:
    """
    Получение массива со значениями пикселей изображения по заданному пути
    :param image_path: путь до данных
    :return: данные в формате BytesIO для подачи на эндпоинт
    """
    byte_img_io = io.BytesIO()
    byte_img = Image.open(image_path)
    byte_img.save(byte_img_io, "PNG")
    byte_img_io.seek(0)
    byte_img = byte_img_io.read()
    files = {"file": ("Test_image.png", byte_img, "multipart/form-data")}
    return files
