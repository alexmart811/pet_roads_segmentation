# Сегментация дорог по изображению со спутника
[![PyTorch - Version](https://img.shields.io/badge/PYTORCH-1.4+-red?style=for-the-badge&logo=pytorch)](https://pepy.tech/project/segmentation-models-pytorch) 
[![Python - Version](https://img.shields.io/badge/PYTHON-3.7+-red?style=for-the-badge&logo=python&logoColor=white)](https://pepy.tech/project/segmentation-models-pytorch)
![](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)



![image info](data/frontend/main_image.png)
## Описание работы
Реализация интерфейса для получения маски дорог с изображения 
        со спутника. В данной работе ставилась задача обучить нейросеть
        находить дороги с изображения, по которой может проехать автомобиль.
        Тестировались модели **Unet** и **DeepLabV3**. Данные для обучения собирались вручную.<br>
## Демо интерфейса
![image info](demo/example.gif)
## Структура проекта
- **backend** <br> 
бэкенд часть
- **config** <br>
конфигурационный файл со значениями переменных
- **data** <br>
Различные данные. images - исходные фотографии со спутника, groundtruth - метки фотографий из папки images.
- **frontend** <br>
фронтенд часть
- **notebooks** <br> 
Jupyter ноутбуки, в которых описана техническая часть: построение моделей, обучение и тестирование
- **report** <br>
Файлы с метриками и значениями функций потерь для их вывода на фронт
