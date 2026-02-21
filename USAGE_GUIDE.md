# Руководство по использованию решения

## 🚀 Быстрый старт (3 шага)

### Шаг 1: Установка зависимостей

```bash
pip install -r requirements.txt
```

### Шаг 2: Запуск решения

**Вариант A - Автоматический режим (рекомендуется):**

```bash
python run_solution.py --mode auto
```

Скрипт автоматически выберет оптимальную конфигурацию на основе размера данных.

**Вариант B - Быстрый baseline:**

```bash
python run_solution.py --mode quick
```

Использует только базовые признаки, работает быстро (~30-60 минут).

**Вариант C - Полный пайплайн:**

```bash
python run_solution.py --mode full
```

Использует все возможности (TSFEL + Lifestream), максимальное качество (~4-6 часов).

### Шаг 3: Проверка результата

После завершения проверьте файл `submission.csv`:

```bash
# Windows PowerShell
Get-Content submission.csv -TotalCount 10

# Linux/Mac
head submission.csv
```

## 📊 Анализ данных (опционально)

Перед обучением можно проанализировать данные:

```bash
python analyze_data.py
```

Это создаст:
- Статистику по последовательностям TCP
- Распределение классов
- Визуализации в папке `plots/`

## ⚙️ Расширенные опции

### Тестирование на малом датасете

```bash
python run_solution.py --mode full --sample-size 10000 --no-lifestream
```

Быстро протестирует пайплайн на 10,000 строк (~5-10 минут).

### Оптимизация по времени

**Быстрый вариант (без Lifestream):**

```bash
python run_solution.py --mode full --no-lifestream
```

Экономит ~50% времени, качество ~97-98%.

**Средний вариант (только статистические TSFEL):**

```bash
python run_solution.py --mode full --no-lifestream --tsfel-domain statistical
```

Экономит ~30% времени, качество ~97-98%.

**Максимальное качество:**

```bash
python run_solution.py --mode full --tsfel-domain all --lifestream-encoder transformer
```

Максимальное время (~6-10 часов), качество ~98-99%.

### Настройка cross-validation

```bash
python run_solution.py --mode full --cv-folds 3
```

Меньше фолдов = быстрее, но менее надежная оценка.

## 🔧 Прямой запуск скриптов

### Quick baseline

```bash
python quick_train.py
```

Простой и быстрый, хороший baseline.

### Полный пайплайн

```bash
python main_pipeline.py --train train.csv --test test.csv --output submission.csv
```

**Параметры:**
- `--sample-size N` - использовать N строк
- `--no-tsfel` - отключить TSFEL
- `--no-lifestream` - отключить Lifestream
- `--tsfel-domain {statistical,temporal,spectral,all}` - домен TSFEL
- `--lifestream-encoder {rnn,transformer}` - тип энкодера
- `--cv-folds N` - количество фолдов CV
- `--chunk-size N` - размер чанка для больших файлов

## 📈 Сравнение подходов

| Подход | Время | Качество | Когда использовать |
|--------|-------|----------|-------------------|
| Quick baseline | ~30-60 мин | 95-97% | Первый запуск, быстрый результат |
| Full без Lifestream | ~2-4 часа | 97-98% | Баланс времени и качества |
| Full с Lifestream (RNN) | ~4-6 часов | 98-99% | Максимальное качество |
| Full с Lifestream (Transformer) | ~6-10 часов | 98-99% | Эксперименты |

## 🐛 Решение проблем

### Ошибка памяти

```bash
# Уменьшите размер чанка
python main_pipeline.py --chunk-size 20000

# Или используйте sample
python main_pipeline.py --sample-size 50000
```

### Долгое обучение

```bash
# Используйте quick режим
python run_solution.py --mode quick

# Или отключите тяжелые компоненты
python run_solution.py --mode full --no-tsfel --no-lifestream
```

### CUDA ошибки

Lifestream автоматически переключится на CPU. Для ускорения:

```bash
# Установите PyTorch с CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Или отключите Lifestream
python run_solution.py --mode full --no-lifestream
```

### TSFEL ошибки

```bash
# Отключите TSFEL
python run_solution.py --mode full --no-tsfel
```

## 📁 Структура файлов после выполнения

```
kaggle_three1/
├── train.csv                    # Исходные данные
├── test.csv
├── sample_submission.csv
│
├── submission.csv              # ✓ Результат!
├── trained_pipeline.pkl        # Сохраненная модель
│
├── plots/                      # Визуализации (если запускали analyze_data.py)
│   ├── sequence_lengths.png
│   ├── packet_sizes.png
│   ├── target_distribution.png
│   └── direction_distribution.png
│
└── [все .py файлы]
```

## 💡 Советы

1. **Первый запуск**: используйте `--mode quick` для быстрого baseline
2. **Тестирование**: используйте `--sample-size 10000` для быстрой проверки
3. **Продакшн**: используйте `--mode full --no-lifestream` для баланса
4. **Максимум**: используйте полный пайплайн только если есть время
5. **Анализ**: запустите `analyze_data.py` для понимания данных

## 🎯 Рекомендуемый workflow

### Для начинающих

```bash
# 1. Анализ данных
python analyze_data.py

# 2. Быстрый baseline
python run_solution.py --mode quick

# 3. Проверка submission
head submission.csv

# 4. Submit to Kaggle
```

### Для опытных

```bash
# 1. Тест на sample
python run_solution.py --mode full --sample-size 10000 --no-lifestream

# 2. Полное обучение
python run_solution.py --mode full --no-lifestream

# 3. Эксперименты с параметрами
python main_pipeline.py --tsfel-domain all --cv-folds 10

# 4. Submit best result
```

## 📞 Поддержка

При возникновении проблем:

1. Проверьте раздел "Решение проблем" выше
2. Запустите с `--sample-size 1000` для отладки
3. Проверьте логи на наличие ошибок
4. Используйте `--no-tsfel --no-lifestream` для минимальной конфигурации

---

**Удачи в соревновании! 🚀**

