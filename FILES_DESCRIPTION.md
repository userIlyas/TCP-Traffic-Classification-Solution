# Описание файлов проекта

## 📁 Структура проекта

### 🎯 Основные исполняемые скрипты

#### `run_solution.py` ⭐ **ГЛАВНЫЙ ФАЙЛ**
**Назначение**: Автоматический запуск решения с выбором оптимальной конфигурации

**Использование**:
```bash
python run_solution.py --mode auto
```

**Возможности**:
- Автоматический выбор конфигурации на основе размера данных
- Оценка времени выполнения
- Проверка результатов
- Три режима: quick, full, auto

---

#### `quick_train.py` ⚡ **БЫСТРЫЙ BASELINE**
**Назначение**: Быстрое обучение с базовыми признаками

**Использование**:
```bash
python quick_train.py
```

**Особенности**:
- Только базовые статистические признаки (~50)
- Быстрое выполнение (30-60 минут)
- Хорошее качество (95-97% accuracy)
- Не требует GPU

---

#### `main_pipeline.py` 🔧 **ПОЛНЫЙ ПАЙПЛАЙН**
**Назначение**: Комплексный пайплайн со всеми возможностями

**Использование**:
```bash
python main_pipeline.py --train train.csv --test test.csv
```

**Возможности**:
- MTU preprocessing
- Target encoding
- TSFEL features
- Lifestream embeddings
- Chunked training
- Cross-validation
- Настраиваемые параметры

---

#### `analyze_data.py` 📊 **АНАЛИЗ ДАННЫХ**
**Назначение**: Исследовательский анализ данных

**Использование**:
```bash
python analyze_data.py
```

**Выходные данные**:
- Статистика последовательностей
- Распределение классов
- Паттерны по классам
- Визуализации в папке `plots/`

---

### 🧩 Модули (библиотеки)

#### `preprocessing.py`
**Содержит**:
- `MTUPreprocessor` - реконструкция данных приложения
- `TargetEncodingFeatures` - target encoding с регуляризацией
- `extract_basic_features()` - базовые статистические признаки

**Ключевые функции**:
- Объединение MTU-пакетов
- Создание категориальных признаков
- Target encoding с CV
- 50+ базовых признаков

---

#### `tsfel_features.py`
**Содержит**:
- `TSFELFeatureExtractor` - извлечение TSFEL признаков
- `CustomTCPFeatures` - кастомные TCP-признаки

**Ключевые функции**:
- Автоматическая генерация 100+ признаков
- Статистические, временные, спектральные домены
- Направленные признаки (отдельно для in/out)
- Burst detection
- Pattern analysis

---

#### `lifestream_embeddings.py`
**Содержит**:
- `TransformerEncoder` - Transformer для последовательностей
- `RNNEncoder` - LSTM для последовательностей
- `ContrastiveLoss` - loss для self-supervised learning
- `LifestreamEmbedder` - главный класс для embeddings

**Ключевые функции**:
- Self-supervised learning (CoLES)
- Два типа энкодеров (RNN/Transformer)
- 128-мерные embeddings
- Contrastive learning

---

#### `chunked_training.py`
**Содержит**:
- `ChunkedDataProcessor` - обработка больших файлов
- `IncrementalLGBMTrainer` - инкрементальное обучение LightGBM

**Ключевые функции**:
- Чанковая загрузка данных
- Incremental training
- Cross-validation
- Ensemble predictions
- Управление памятью

---

### 📚 Документация

#### `README.md` 📖
**Содержание**:
- Описание задачи
- Архитектура решения
- Инструкции по установке
- Примеры использования
- Технические детали
- Troubleshooting

**Для кого**: Все пользователи

---

#### `USAGE_GUIDE.md` 🎓
**Содержание**:
- Быстрый старт (3 шага)
- Расширенные опции
- Сравнение подходов
- Решение проблем
- Рекомендуемый workflow

**Для кого**: Начинающие пользователи

---

#### `SOLUTION_SUMMARY.md` 🔬
**Содержание**:
- Научное обоснование
- Технические детали
- Feature engineering pipeline
- Ожидаемые результаты
- Lessons learned
- Рекомендации по улучшению

**Для кого**: Опытные пользователи, исследователи

---

#### `COMMANDS.txt` ⌨️
**Содержание**:
- Быстрый справочник команд
- Примеры использования
- Параметры
- Оценки времени

**Для кого**: Быстрая справка

---

#### `FILES_DESCRIPTION.md` 📁 (этот файл)
**Содержание**:
- Описание всех файлов проекта
- Назначение каждого файла
- Примеры использования

**Для кого**: Навигация по проекту

---

### 📦 Конфигурация

#### `requirements.txt`
**Содержание**: Список зависимостей Python

**Установка**:
```bash
pip install -r requirements.txt
```

**Зависимости**:
- pandas, numpy - обработка данных
- scikit-learn - ML утилиты
- lightgbm - gradient boosting
- category-encoders - target encoding
- tsfel - time-series features
- torch - deep learning (Lifestream)
- tqdm - progress bars
- matplotlib, seaborn - визуализация

---

### 📊 Данные

#### `train.csv` (исходные данные)
**Размер**: ~1.4 GB  
**Формат**: CSV  
**Колонки**:
- `app_service` - целевая переменная (класс приложения)
- `tcp_len_1` ... `tcp_len_30` - длины TCP пакетов

#### `test.csv` (исходные данные)
**Размер**: ~830 MB  
**Формат**: CSV  
**Колонки**:
- `id` - идентификатор
- `tcp_len_1` ... `tcp_len_30` - длины TCP пакетов

#### `sample_submission.csv` (исходные данные)
**Размер**: 116 B  
**Формат**: Пример submission файла

#### `math_model.pdf` (исходные данные)
**Размер**: 1.3 MB  
**Содержание**: Математическое описание задачи

---

### 🎯 Выходные файлы (генерируются)

#### `submission.csv` ✅
**Создается**: После выполнения любого скрипта обучения  
**Формат**:
```csv
id,app_service
0,telegram
1,http
...
```

**Использование**: Submit to Kaggle

---

#### `trained_pipeline.pkl`
**Создается**: `main_pipeline.py`  
**Содержание**: Сохраненная модель и все компоненты  
**Размер**: ~100-200 MB

**Загрузка**:
```python
from main_pipeline import TCPTrafficClassifier
classifier = TCPTrafficClassifier()
classifier.load_pipeline('trained_pipeline.pkl')
```

---

#### `plots/` (папка)
**Создается**: `analyze_data.py`  
**Содержание**:
- `sequence_lengths.png` - распределение длин последовательностей
- `packet_sizes.png` - распределение размеров пакетов
- `target_distribution.png` - распределение классов
- `direction_distribution.png` - распределение направлений

---

## 🎯 Рекомендуемый порядок изучения

### Для начинающих:
1. `README.md` - общее понимание
2. `USAGE_GUIDE.md` - как использовать
3. `COMMANDS.txt` - быстрая справка
4. `run_solution.py` - запуск решения

### Для опытных:
1. `SOLUTION_SUMMARY.md` - техническое описание
2. `preprocessing.py` - feature engineering
3. `tsfel_features.py` - time-series features
4. `lifestream_embeddings.py` - deep learning
5. `main_pipeline.py` - полный пайплайн

### Для исследователей:
1. `SOLUTION_SUMMARY.md` - научное обоснование
2. Все модули (`preprocessing.py`, `tsfel_features.py`, etc.)
3. `analyze_data.py` - анализ данных
4. Экспериментируйте с параметрами!

---

## 📈 Зависимости между файлами

```
run_solution.py
    ├─→ quick_train.py (mode=quick)
    │       └─→ [прямая работа с данными]
    │
    └─→ main_pipeline.py (mode=full)
            ├─→ preprocessing.py
            │       └─→ MTU, Target Encoding, Basic Features
            ├─→ tsfel_features.py
            │       └─→ TSFEL, Custom TCP Features
            ├─→ lifestream_embeddings.py
            │       └─→ RNN/Transformer Embeddings
            └─→ chunked_training.py
                    └─→ LightGBM Training

analyze_data.py
    └─→ [прямая работа с данными]
```

---

## 🔧 Модификация и расширение

### Добавление новых признаков
**Файл**: `preprocessing.py`  
**Функция**: `extract_basic_features()`

### Изменение TSFEL конфигурации
**Файл**: `tsfel_features.py`  
**Класс**: `TSFELFeatureExtractor.__init__()`

### Настройка Lifestream
**Файл**: `lifestream_embeddings.py`  
**Классы**: `TransformerEncoder`, `RNNEncoder`

### Изменение параметров LightGBM
**Файл**: `chunked_training.py`  
**Метод**: `IncrementalLGBMTrainer._get_default_params()`

---

## 💡 Советы

1. **Начните с `run_solution.py --mode auto`** - автоматически выберет лучший вариант
2. **Используйте `COMMANDS.txt`** для быстрого поиска команд
3. **Читайте `README.md`** для общего понимания
4. **Изучайте `SOLUTION_SUMMARY.md`** для глубокого понимания
5. **Экспериментируйте с `main_pipeline.py`** для кастомизации

---

## 📞 Поддержка

При возникновении вопросов:
1. Проверьте `README.md` → раздел Troubleshooting
2. Посмотрите `USAGE_GUIDE.md` → раздел "Решение проблем"
3. Изучите `COMMANDS.txt` → примеры команд
4. Запустите с `--sample-size 1000` для отладки

---

**Удачи в соревновании! 🚀**

