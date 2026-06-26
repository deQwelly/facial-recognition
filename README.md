# Facial Recognition (PyTorch)

## Описание проекта
Система распознавания лиц на основе глубинного метрического обучения. Реализовано сравнение изображений через эмбеддинги и Triplet Loss.

## Цель
Построение модели, способной эффективно различать изображения разных людей при малом объёме данных (~1000 изображений на обучении).

## Подход
- Backbone: ResNet18 (ImageNet pretrained)
- Метод: Metric Learning
- Loss: Triplet Margin Loss
- Mining: Hard Triplet Mining
- Sampler: MPerClassSampler
- Оценка качества: ROC-AUC по расстояниям между эмбеддингами

## Предобработка данных
- Resize и нормализация по статистике датасета
- Аугментации:
  - Random Horizontal Flip
  - Color Jitter (brightness, contrast, saturation)

## Архитектура пайплайна
- Извлечение эмбеддингов через CNN
- Обучение с Triplet Loss
- Формирование пар для валидации и тестирования
- Оценка через ROC-AUC на pairwise distance

## Результаты
- Проведён подбор гиперпараметров (learning rate, weight decay) с использованием grid search.
- Использованы техники hard triplet mining и MPerClassSampler для стабилизации обучения metric learning модели.
- Оценка качества выполнялась по метрике ROC-AUC на попарных расстояниях эмбеддингов.
- __ROC-AUC = 0.99__

## Стек технологий
- Python
- PyTorch
- torchvision
- scikit-learn
- pytorch-metric-learning
- Weights & Biases

## Особенности реализации
- MPerClassSampler для сбалансированных батчей
- Hard triplet mining
- Подбор гиперпараметров (learning rate, weight decay) через grid search
- Использование early stopping по validation loss
- Сохранение лучшей модели по ROC-AUC
- Полный цикл экспериментов с логированием в W&B
