"""
Программа: Отрисовка графиков
Версия: 1.0
"""

from typing import List
import matplotlib
import matplotlib.pyplot as plt


def model_graph(
    tr: List[float], val: List[float], epochs: int, loss: bool, title: str
) -> matplotlib.figure.Figure:
    """
    отрисовка графиков потерь и метрик
    :params tr: список потерь/метрики на трейне
    :params val: список потерь/метрики на тесте
    :params epochs: количество эпох
    :params loss: флаг графика потерь (True - отрисовка потерь,
                  False - отрисовка метрик)
    :params title: название графика
    :return: поле рисунка
    """
    fig = plt.figure(figsize=(15, 7))
    plt.title(title)
    plt.plot(range(epochs), tr, label="train", linewidth=2)
    plt.plot(range(epochs), val, label="val", linewidth=2)
    if loss:
        plt.ylim([0, 1])
    plt.grid()
    plt.xlabel("epoches")
    plt.legend()
    return fig


def show_scores(epochs: int, **graphs) -> matplotlib.figure.Figure:
    """
    Отрисовка метрики всех моделей на тесте
    :params epochs: количество эпох
    :params graphs: словари потерь и метрик моделей
    :return: поле рисунка
    """
    fig = plt.figure(figsize=(15, 7))
    plt.title("Метрики моделей на валидации")
    for name, graph in graphs.items():
        plt.plot(range(epochs), graph["val_scores"], label=f"{name}"[:-5], 
                 linewidth=2)
    plt.grid()
    plt.xlabel("epoches")
    plt.legend()
    return fig
