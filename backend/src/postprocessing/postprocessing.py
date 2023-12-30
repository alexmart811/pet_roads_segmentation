"""
Программа: Обработка изображения перед сохранением ее по пути в директорию
Версия: 1.0
"""

import numpy as np
from PIL import Image
from typing import Text


def postproc_mask(im: np.ndarray, save_path: Text) -> None:
    """
    Обработка и сохранение изображения
    :params im: массив изображения
    :params save_path: путь до сохраненного изображения
    """
    im = im[0][0].astype("uint8") * 255
    image_png = Image.fromarray(im)
    image_png.save(save_path)
