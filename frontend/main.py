"""
Программа: Frontend часть проекта
Версия: 1.0
"""

import os
import yaml
import streamlit as st
from src.plotting.charts import model_graph, show_scores
from src.data.get_data import from_pickle, get_image
from src.evaluate.evaluate import evaluate_from_file, show_out_mask

CONFIG_PATH = "../config/params.yml"


def main_page():
    """
    Страница с описанием проекта
    """
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    st.markdown("# Сегментация дорог по изображению со спутника")
    st.image(config["frontend"]["main_image"], width=900)
    st.write(
        """
        Реализация интерфейса для получения маски дорог с изображения 
        со спутника. В данной работе ставилась задача обучить нейросеть
        находить дороги с изображения, по которой может проехать автомобиль.
        Тестировались модели Unet и DeepLabV3, графики метрик которых можно 
        найти во вкладке с анализом. Данные для обучения собирались вручную."""
    )


def exploratory():
    """
    Exploratory data analysis
    """
    st.markdown("# Метрики и лоссы получившихся моделей")

    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    # создание чекбоксов
    st.sidebar.markdown("# Выбор графиков")
    unet_bce = st.sidebar.checkbox("unet bce")
    unet_dice = st.sidebar.checkbox("unet dice")
    unet_focal = st.sidebar.checkbox("unet focal")
    unet_lova = st.sidebar.checkbox("unet lova")
    deeplab_bce = st.sidebar.checkbox("deeplab bce")
    deeplab_dice = st.sidebar.checkbox("deeplab dice")
    deeplab_focal = st.sidebar.checkbox("deeplab focal")
    deeplab_lova = st.sidebar.checkbox("deeplab lova")
    general_scores = st.sidebar.checkbox("general scores")

    # Считывание словарей с данными из файла
    unet_bce_dict = from_pickle(config["train"]["unet_bce_dict_path"])
    unet_dice_dict = from_pickle(config["train"]["unet_dice_dict_path"])
    unet_focal_dict = from_pickle(config["train"]["unet_focal_dict_path"])
    unet_lova_dict = from_pickle(config["train"]["unet_lova_dict_path"])
    deeplab_bce_dict = from_pickle(config["train"]["deeplab_bce_dict_path"])
    deeplab_dice_dict = from_pickle(config["train"]["deeplab_dice_dict_path"])
    deeplab_focal_dict = from_pickle(config["train"]["deeplab_focal_dict_path"])
    deeplab_lova_dict = from_pickle(config["train"]["deeplab_lova_dict_path"])
    epochs = config["train"]["epochs"]

    if unet_bce:
        st.markdown("### Unet bce")
        st.pyplot(
            model_graph(
                unet_bce_dict["train_losses"],
                unet_bce_dict["val_losses"],
                epochs,
                True,
                "Unet bce loss",
            )
        )
        st.pyplot(
            model_graph(
                unet_bce_dict["train_scores"],
                unet_bce_dict["val_scores"],
                epochs,
                False,
                "Unet bce scores",
            )
        )
    if unet_dice:
        st.markdown("### Unet dice")
        st.pyplot(
            model_graph(
                unet_dice_dict["train_losses"],
                unet_dice_dict["val_losses"],
                epochs,
                True,
                "Unet dice loss",
            )
        )
        st.pyplot(
            model_graph(
                unet_dice_dict["train_scores"],
                unet_dice_dict["val_scores"],
                epochs,
                False,
                "Unet dice scores",
            )
        )
    if unet_focal:
        st.markdown("### Unet focal")
        st.pyplot(
            model_graph(
                unet_focal_dict["train_losses"],
                unet_focal_dict["val_losses"],
                epochs,
                True,
                "Unet focal loss",
            )
        )
        st.pyplot(
            model_graph(
                unet_focal_dict["train_scores"],
                unet_focal_dict["val_scores"],
                epochs,
                False,
                "Unet focal scores",
            )
        )
    if unet_lova:
        st.markdown("### Unet lova")
        st.pyplot(
            model_graph(
                unet_lova_dict["train_losses"],
                unet_lova_dict["val_losses"],
                epochs,
                False,
                "Unet lova loss",
            )
        )
        st.pyplot(
            model_graph(
                unet_lova_dict["train_scores"],
                unet_lova_dict["val_scores"],
                epochs,
                False,
                "Unet lova scores",
            )
        )
    if deeplab_bce:
        st.markdown("### deeplab bce")
        st.pyplot(
            model_graph(
                deeplab_bce_dict["train_losses"],
                deeplab_bce_dict["val_losses"],
                epochs,
                True,
                "DeepLab bce loss",
            )
        )
        st.pyplot(
            model_graph(
                deeplab_bce_dict["train_scores"],
                deeplab_bce_dict["val_scores"],
                epochs,
                False,
                "DeepLab bce scores",
            )
        )
    if deeplab_dice:
        st.markdown("### deeplab dice")
        st.pyplot(
            model_graph(
                deeplab_dice_dict["train_losses"],
                deeplab_dice_dict["val_losses"],
                epochs,
                False,
                "DeepLab dice loss",
            )
        )
        st.pyplot(
            model_graph(
                deeplab_dice_dict["train_scores"],
                deeplab_dice_dict["val_scores"],
                epochs,
                False,
                "DeepLab dice scores",
            )
        )
    if deeplab_focal:
        st.markdown("### deeplab focal")
        st.pyplot(
            model_graph(
                deeplab_focal_dict["train_losses"],
                deeplab_focal_dict["val_losses"],
                epochs,
                True,
                "DeepLab focal loss",
            )
        )
        st.pyplot(
            model_graph(
                deeplab_focal_dict["train_scores"],
                deeplab_focal_dict["val_scores"],
                epochs,
                False,
                "DeepLab focal scores",
            )
        )
    if deeplab_lova:
        st.markdown("### deeplab lova")
        st.pyplot(
            model_graph(
                deeplab_lova_dict["train_losses"],
                deeplab_lova_dict["val_losses"],
                epochs,
                False,
                "DeepLab lova loss",
            )
        )
        st.pyplot(
            model_graph(
                deeplab_lova_dict["train_scores"],
                deeplab_lova_dict["val_scores"],
                epochs,
                False,
                "DeepLab lova scores",
            )
        )
    if general_scores:
        st.markdown("### Графики метрик всех моделей")
        st.pyplot(
            show_scores(
                epochs,
                unet_bce_dict=unet_bce_dict,
                unet_dice_dict=unet_dice_dict,
                unet_focal_dict=unet_focal_dict,
                unet_lova_dict=unet_lova_dict,
                deeplab_bce_dict=deeplab_bce_dict,
                deeplab_dice_dict=deeplab_dice_dict,
                deeplab_focal_dict=deeplab_focal_dict,
                deeplab_lova_dict=deeplab_lova_dict,
            )
        )
        st.markdown("### Общие выводы")
        st.write("""У модели Unet на всех функциях потерь переобучение 
                 минимальное, но метрика хуже, в отличие от DeepLab.
                 По итогу была выбрана модель DeepLab с функцией потерь bce, 
                 так как у этой модели скор наибольший, хоть и наблюдается 
                 переобучение примерно в 0.2 единиц.""")


def prediction_from_file():
    """
    Получение предсказаний из файла с данными
    """
    st.markdown("# Prediction")
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    endpoint = config["endpoints"]["prediction_from_file"]
    out_mask_path = config["eval"]["out_mask"]

    # Загрузка изображения
    upload_file = st.file_uploader("", type=["png"])
    if upload_file:
        st.markdown("## Вход")
        # Вывод изображения на экран и подача его на эндпоинт
        st.image(upload_file, width=300)
        files = get_image(image_path=upload_file)
        evaluate_from_file(endpoint=endpoint, files=files)
        # Вывод получившейся маски на экран
        if os.path.exists(out_mask_path):
            st.markdown("## Выход")
            show_out_mask(out_mask_path)


def main():
    """
    Сборка пайплайна в одном блоке
    """
    with open(CONFIG_PATH, encoding="utf-8") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    out_mask_path = config["eval"]["out_mask"]
    try:
        os.remove(out_mask_path)
    except FileNotFoundError:
        pass
    page_names_to_funcs = {
        "Описание проекта": main_page,
        "Графики метрик и лоссов": exploratory,
        "Тест на собственном изображении": prediction_from_file,
    }
    selected_page = st.sidebar.selectbox("Выберите пункт", 
                                         page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()


if __name__ == "__main__":
    main()
