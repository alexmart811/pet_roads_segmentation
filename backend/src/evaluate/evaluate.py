"""
Программа: Получение маски изображения
Версия: 1.0
"""

import yaml
import torch
import numpy as np
from typing import Text, Any

from ..data.get_data import get_image, image_preproc, deeplab_trained
from ..postprocessing.postprocessing import postproc_mask


def pipeline_evaluate(config_path: Text, data_path: Text) -> None:
    """
    Предобработка входных данных. Запись получившейся маски по пути.
    :param config_path: путь до конфигурационного файла
    :param data_path: путь до файла с данными
    """
    with open(config_path, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train_config = config["train"]
    eval_config = config["eval"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Получение массива данных
    im = get_image(image_path=data_path)

    size = train_config["size"]
    # Препроцессинг изображения для подачи в сетку
    im = image_preproc(im, size, device)

    model_weights_path = train_config["deeplab_bce_weights_path"]
    # Загрузка обученной модели
    deeplab_model = deeplab_trained(model_weights_path, device)

    # Получение маски из изображения
    out = evaluate_model_on_image(deeplab_model, im)

    out_image_path = eval_config["out_mask"]
    # Сохранение изображения
    postproc_mask(out, out_image_path)


def evaluate_model_on_image(model: Any, image: torch.Tensor) -> np.ndarray:
    """
    Получение маски изображения по введенной модели
    :params model: объект модели
    :params image: массив изображения
    :return: маска изображения
    """
    model.eval()
    with torch.no_grad():
        out = model(image)
        out = (out.to("cpu") > 0.5).type(torch.long).detach().numpy()
        return out
