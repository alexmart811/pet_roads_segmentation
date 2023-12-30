"""
Программа: Получение данных из файла, предобработка данных
Версия: 1.0
"""

from typing import Text, Tuple
import numpy as np
import cv2
import torch
from skimage.io import imread
from skimage.transform import resize
import segmentation_models_pytorch as smp


def get_image(image_path: Text) -> np.ndarray:
    """
    Получение массива со значениями пикселей изображения по заданному пути
    :param image_path: путь до данных
    :return: массив изображения
    """
    return imread(image_path)


def image_preproc(
    im: np.ndarray, size: Tuple[int], device: Text
) -> torch.Tensor:
    """
    Предобработка входного изображения
    :params im: массив входного изображения
    :params size: разрешение, к которому нужно привести изображение
    :params device: gpu или cpu
    :return: предобработанное изображение
    """
    im = resize(
        cv2.cvtColor(im, cv2.COLOR_BGRA2BGR), size, mode="constant", 
        anti_aliasing=True
    )
    im = np.array(im, np.float32)
    im = np.rollaxis(im, 2, 0)
    im = im.reshape((1, 3, size[0], size[1]))
    im = torch.Tensor(im).to(device)
    return im


def deeplab_trained(
    model_path: Text, device: Text
) -> smp.decoders.deeplabv3.model.DeepLabV3Plus:
    """
    Получение предобученной модели deeplab
    :params model_path: путь до весов модели
    :params device: gpu или cpu
    :return: объект модели
    """
    deeplab_model = smp.DeepLabV3Plus(
        encoder_name="resnet101", encoder_weights="imagenet",
        classes=1, activation=None
    ).to(device)
    deeplab_model.load_state_dict(torch.load(model_path, 
                                             map_location=torch.device(device)))
    return deeplab_model
