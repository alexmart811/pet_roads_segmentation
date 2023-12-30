"""
Программа: подача данных на бекенд, а также считывание с файла и отрисовка
интерфейса сайта
Версия: 1.0
"""
from typing import Text
import io
import streamlit as st
from skimage.io import imread
import requests


def evaluate_from_file(endpoint: Text, files: io.BytesIO) -> None:
    """
    Подача данных на эндпоинт в бекенд
    :param endpoint: endpoint
    :param files: файл в формате BytesIO
    """
    button_ok = st.button("Predict")
    if button_ok:
        with st.spinner("Модель обрабатывает изображение..."):
            requests.post(endpoint, files=files)


def show_out_mask(out_mask_path: Text) -> None:
    """
    Считывание изображения с файла и вывод на экран streamlit
    :params out_mask_path: путь до изображения
    """
    out = imread(out_mask_path)
    st.image(out, width=300)
    with open(out_mask_path, "rb") as file:
        st.download_button(label="Загрузить маску", data=file, file_name="mask.png")
