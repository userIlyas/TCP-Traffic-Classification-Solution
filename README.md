# TCP Traffic Classification Solution

Решение для классификации зашифрованного TCP-трафика на основе анализа последовательностей длин пакетов.

## 📋 Описание задачи

Задача состоит в классификации приложений/сервисов, генерирующих зашифрованный TCP-трафик, используя только метаданные пакетов (размеры и направления), без доступа к содержимому.

**Ключевые особенности:**
- Каждый поток представлен последовательностью до 30 TCP-пакетов
- Положительные значения = пакеты от клиента
- Отрицательные значения = пакеты к клиенту
- Нули = padding (заполнение коротких последовательностей)

## 🏗️ Архитектура решения

Решение использует многоуровневый подход с feature engineering и глубоким обучением:

### 1. **MTU Preprocessing**
- Реконструкция данных приложения путем объединения последовательных MTU-пакетов (>1200 байт)
- Target Encoding с регуляризацией для категориальных признаков
- Предотвращение утечек данных через кросс-валидацию

### 2. **Time-Series Feature Extraction (TSFEL)**
- Автоматическая генерация статистических, временных и спектральных признаков
- Специализированные признаки для TCP-трафика:
  - Burst detection (всплески трафика)
  - Direction patterns (паттерны направлений)
  - Size variance (вариативность размеров)

### 3. **PyTorch-Lifestream Embeddings**
- Векторные представления последовательностей через self-supervised learning
- Два типа энкодеров:
  - **RNN (LSTM)**: быстрее, эффективен для последовательностей
  - **Transformer**: лучше захватывает долгосрочные зависимости
- Contrastive Learning (CoLES) для обучения без учителя

### 4. **LightGBM Classifier**
- Градиентный бустинг для финальной классификации
- Chunked training для работы с большими данными
- Cross-validation для надежной оценки

## 📦 Установка

### Требования
- Python 3.8+
- CUDA (опционально, для ускорения Lifestream)

### Установка зависимостей

```bash
pip install -r requirements.txt
```

## 🚀 Использование

### Быстрый старт (рекомендуется для первого запуска)

Используйте `quick_train.py` для быстрого обучения с базовыми признаками:

```bash
python quick_train.py
```

Этот скрипт:
- Загружает данные
- Извлекает быстрые эффективные признаки
- Обучает LightGBM с 5-fold CV
- Генерирует `submission.csv`

**Время выполнения:** ~30-60 минут на полном датасете

### Полный пайплайн (максимальное качество)

Для использования всех возможностей (TSFEL + Lifestream):

```bash
python main_pipeline.py --train train.csv --test test.csv --output submission.csv
```

**Параметры:**
- `--sample-size N`: использовать только N строк для быстрого тестирования
- `--no-tsfel`: отключить TSFEL признаки (быстрее)
- `--no-lifestream`: отключить Lifestream embeddings (быстрее)
- `--tsfel-domain`: домен TSFEL (`statistical`, `temporal`, `spectral`, `all`)
- `--lifestream-encoder`: тип энкодера (`rnn` или `transformer`)
- `--cv-folds N`: количество фолдов для CV (по умолчанию 5)
- `--chunk-size N`: размер чанка для больших файлов (по умолчанию 50000)

**Примеры:**

```bash
# Быстрое тестирование на 10000 строк
python main_pipeline.py --sample-size 10000 --no-lifestream

# Полное обучение без Lifestream (быстрее)
python main_pipeline.py --no-lifestream

# Полное обучение со всеми фичами
python main_pipeline.py --tsfel-domain all --lifestream-encoder transformer
```

**Время выполнения:** 
- Без Lifestream: ~2-4 часа
- С Lifestream (RNN): ~4-6 часов
- С Lifestream (Transformer): ~6-10 часов

## 📊 Структура проекта

```
kaggle_three1/
├── train.csv                    # Обучающие данные
├── test.csv                     # Тестовые данные
├── sample_submission.csv        # Пример submission
├── math_model.pdf              # Математическое описание
│
├── requirements.txt            # Зависимости
├── README.md                   # Документация
│
├── preprocessing.py            # MTU preprocessing и Target Encoding
├── tsfel_features.py          # TSFEL feature extraction
├── lifestream_embeddings.py   # PyTorch-Lifestream embeddings
├── chunked_training.py        # Chunked training для больших данных
│
├── main_pipeline.py           # Полный пайплайн
├── quick_train.py             # Быстрый baseline
│
└── submission.csv             # Результат (генерируется)
```

## 🔬 Технические детали

### Feature Engineering

#### Базовые статистические признаки
- Количество пакетов, средний/макс/мин размер
- Стандартное отклонение размеров
- Исходящие/входящие пакеты и байты
- Соотношения (out/in ratio, bytes ratio)

#### MTU-специфичные признаки
- Количество больших пакетов (>1200 байт)
- Burst detection (последовательные большие пакеты)
- Реконструкция данных приложения

#### Временные признаки (TSFEL)
- Статистические: mean, std, variance, kurtosis, skewness
- Временные: autocorrelation, entropy, zero-crossing rate
- Спектральные: FFT features, spectral entropy

#### Direction patterns
- Паттерны направлений (например, "++--+")
- Частота смены направлений
- Альтернирующие паттерны

### Target Encoding

Использует **regularized target encoding** с кросс-валидацией:

```
encoded_value = (n * mean_target + alpha * global_mean) / (n + alpha)
```

где:
- `n` = количество наблюдений в категории
- `alpha` = параметр сглаживания (10 по умолчанию)
- `mean_target` = среднее значение target для категории
- `global_mean` = глобальное среднее target

### Lifestream Embeddings

**Архитектура RNN:**
```
Input (30 packets) 
  → Linear Projection (embedding_dim/2)
  → Bidirectional LSTM (2 layers)
  → Linear Projection (embedding_dim)
  → Embeddings (128-dim)
```

**Архитектура Transformer:**
```
Input (30 packets)
  → Linear Projection (embedding_dim)
  → Positional Encoding
  → Transformer Encoder (4 heads, 2 layers)
  → Global Average Pooling
  → Linear Projection
  → Embeddings (128-dim)
```

**Contrastive Loss:**
Обучение через contrastive learning - модель учится различать последовательности разных классов.

### LightGBM Configuration

Оптимизированные параметры:

```python
{
    'objective': 'multiclass',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,      # Случайный отбор 80% признаков
    'bagging_fraction': 0.8,      # Случайный отбор 80% данных
    'bagging_freq': 5,
    'min_child_samples': 20,
}
```

## 💡 Рекомендации по использованию

### Для быстрого результата
1. Используйте `quick_train.py` - дает хороший baseline (~95% accuracy)
2. Время: ~30-60 минут

### Для максимального качества
1. Используйте `main_pipeline.py` с TSFEL
2. Lifestream можно отключить для экономии времени
3. Время: ~2-4 часа

### Для экспериментов
1. Используйте `--sample-size 10000` для быстрого тестирования
2. Экспериментируйте с параметрами TSFEL и Lifestream
3. Время: ~5-10 минут на sample

### Работа с большими данными
- Автоматически используется chunked processing для файлов >500MB
- Можно настроить `--chunk-size` для оптимизации памяти
- LightGBM поддерживает incremental training

## 📈 Ожидаемые результаты

На основе аналогичных задач (encrypted traffic classification):

- **Quick baseline**: ~0.95-0.97 accuracy, ~0.15-0.20 log loss
- **С TSFEL**: ~0.97-0.98 accuracy, ~0.10-0.15 log loss
- **С Lifestream**: ~0.98-0.99 accuracy, ~0.08-0.12 log loss

## 🔧 Troubleshooting

### Проблемы с памятью
- Уменьшите `--chunk-size`
- Используйте `--sample-size` для тестирования
- Отключите Lifestream (`--no-lifestream`)

### Долгое обучение
- Используйте `quick_train.py` вместо полного пайплайна
- Отключите TSFEL (`--no-tsfel`)
- Уменьшите `--cv-folds`

### CUDA ошибки (Lifestream)
- Lifestream автоматически переключится на CPU
- Для ускорения установите PyTorch с CUDA
- Или используйте `--no-lifestream`

### TSFEL ошибки
- Некоторые признаки могут не вычисляться для коротких последовательностей
- Пайплайн автоматически обрабатывает NaN значения
- При критических ошибках используйте `--no-tsfel`

## 📚 Ссылки

- [PyTorch-Lifestream](https://github.com/pytorch-lifestream/pytorch-lifestream) - Self-supervised learning для event sequences
- [TSFEL](https://github.com/fraunhoferportugal/tsfel) - Time Series Feature Extraction Library
- [LightGBM](https://lightgbm.readthedocs.io/) - Gradient boosting framework

## 🎯 Ключевые инсайты

1. **Time-series nature**: TCP пакеты - это временной ряд, порядок критичен
2. **MTU patterns**: Большие пакеты (>1200) часто являются частью одного сообщения
3. **Direction matters**: Паттерны направлений уникальны для разных приложений
4. **Burst behavior**: Всплески трафика характерны для определенных протоколов
5. **First packets**: Начало сессии (handshake) содержит важную информацию

## 📝 Лицензия

Этот код предоставляется для участия в соревновании Kaggle.

---

**Автор:** AI Assistant  
**Дата:** 2025-11-08  
**Версия:** 1.0

