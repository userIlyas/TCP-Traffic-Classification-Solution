# 📚 Гайд: Автоматическая синхронизация checkpoints между аккаунтами Google Colab

## 🎯 Что было добавлено

1. **Автоматическая синхронизация Google Drive** - код автоматически скачивает папку checkpoints из общей Google Drive папки
2. **GPU поддержка для LightGBM** - автоматическое использование GPU если доступно
3. **Resume логика** - автоматическое продолжение обучения с места остановки
4. **Логирование времени** - показывает время обучения каждого батча

## 🚀 Как использовать

### Шаг 1: Настройка общей папки Google Drive

1. Создайте папку `checkpoints` в вашем Google Drive (любой аккаунт)
2. Сделайте папку **доступной по ссылке** (правой кнопкой → "Настроить доступ" → "Все, у кого есть ссылка")
3. Скопируйте ID папки из ссылки

   Например, если ссылка: `https://drive.google.com/drive/folders/1EwJxSVjhLYD_2c1z2HDMCRRgyqux0tlv`
   
   То ID = `1EwJxSVjhLYD_2c1z2HDMCRRgyqux0tlv`

4. В файле `main.py` найдите строку с `FOLDER_ID` и замените на ваш ID:
   ```python
   FOLDER_ID = '1EwJxSVjhLYD_2c1z2HDMCRRgyqux0tlv'  # Замените на ваш ID
   ```

### Шаг 2: Запуск в Colab

1. Откройте Google Colab (любой аккаунт)
2. Загрузите файл `main.py` в Colab
3. Запустите код - он автоматически:
   - Смонтирует ваш Google Drive
   - Проверит наличие папки `/content/drive/MyDrive/checkpoints`
   - Если папка пустая или не существует - скачает файлы из общей папки
   - Если файлы уже есть - пропустит скачивание

### Шаг 3: Обучение с resume

Код автоматически проверяет наличие обученных моделей:

- **Если модель уже обучена** (`model_fold_X_bag_Y.pkl` существует):
  - Загружает модель из файла
  - Пропускает обучение этого батча
  - Продолжает со следующего батча

- **Если модель не найдена**:
  - Обучает модель
  - Сохраняет сразу после обучения
  - Показывает время обучения

### Шаг 4: Синхронизация обратно в общую папку

После обучения на новом аккаунте:

1. **Вариант 1 (Ручной):**
   - Откройте Google Drive веб-интерфейс
   - Перейдите в `/content/drive/MyDrive/checkpoints`
   - Скопируйте новые `.pkl` файлы в общую папку

2. **Вариант 2 (Автоматический - требует настройки):**
   - Используйте Google Drive API с сервисным аккаунтом
   - Или используйте `rclone` / `gdrive` CLI инструменты

## ⚙️ Технические детали

### GPU поддержка

Код автоматически определяет доступность GPU и настраивает LightGBM:

```python
if torch.cuda.is_available():
    params['device'] = 'gpu'
    params['gpu_platform_id'] = 0
    params['gpu_device_id'] = 0
```

**Важно:** LightGBM требует установки GPU версии:
```bash
!pip install lightgbm --upgrade
```

### Структура файлов checkpoints

```
checkpoints/
├── chunk_index.pkl                    # Индекс обработанных чанков
├── lifestream_embeddings_index.pkl   # Индекс эмбеддингов
├── test_chunk_index.pkl               # Индекс тестовых чанков
├── test_embeddings_index.pkl         # Индекс тестовых эмбеддингов
├── trained_models.pkl                 # Финальные обученные модели
├── model_fold_1_bag_1.pkl            # Модель fold 1, batch 1
├── model_fold_1_bag_2.pkl            # Модель fold 1, batch 2
├── model_fold_2_bag_1.pkl            # Модель fold 2, batch 1
└── ...                                # И так далее
```

### Resume логика

При запуске код проверяет:

1. **Существование файла модели:**
   ```python
   model_path = f'{CHECKPOINT_DIR}/model_fold_{fold_num}_bag_{batch_num}.pkl'
   if os.path.exists(model_path):
       model = joblib.load(model_path)  # Загружаем
       continue  # Пропускаем обучение
   ```

2. **Если файл поврежден:**
   - Код попытается переобучить модель
   - Покажет предупреждение

## 🔧 Настройка

### Изменение ID папки Google Drive

В файле `main.py`, строка ~119:
```python
FOLDER_ID = '1EwJxSVjhLYD_2c1z2HDMCRRgyqux0tlv'  # Замените на ваш ID
```

### Изменение пути к checkpoints

По умолчанию: `/content/drive/MyDrive/checkpoints`

Для изменения найдите строку ~118 в `main.py`:
```python
CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoints'
```

## 📊 Логирование

Код показывает:

- ✅ Успешную синхронизацию
- ⚠️ Предупреждения (если синхронизация не удалась)
- 📥 Процесс скачивания файлов
- ⏱️ Время обучения каждого батча: `✓ (45.2s)`
- 📁 Количество найденных файлов в папке

## 🐛 Решение проблем

### Проблема: "Failed to sync from Google Drive"

**Решение:**
1. Проверьте, что папка доступна по ссылке
2. Проверьте правильность FOLDER_ID
3. Убедитесь, что установлен `gdown`: `!pip install -U gdown`

### Проблема: "GPU not available"

**Решение:**
1. В Colab: Runtime → Change runtime type → GPU
2. Проверьте: `torch.cuda.is_available()` должно вернуть `True`
3. Установите GPU версию LightGBM: `!pip install lightgbm --upgrade`

### Проблема: Модели не сохраняются

**Решение:**
1. Проверьте права доступа к Google Drive
2. Убедитесь, что диск смонтирован: `ls /content/drive/MyDrive/`
3. Проверьте свободное место на диске

### Проблема: Resume не работает

**Решение:**
1. Проверьте, что файлы действительно существуют: `ls /content/drive/MyDrive/checkpoints/`
2. Убедитесь, что имена файлов соответствуют формату: `model_fold_X_bag_Y.pkl`
3. Проверьте логи - возможно файл поврежден и требуется переобучение

## 💡 Советы по оптимизации

1. **Используйте GPU** - ускоряет обучение в 2-10 раз
2. **Уменьшите `num_boost_round`** - если обучение слишком долгое
3. **Увеличьте `batch_size_chunks`** - если памяти достаточно
4. **Сохраняйте после каждого батча** - уже реализовано! ✅

## 📝 Пример вывода

```
======================================================================
STEP 2: Setting up Google Drive sync...
======================================================================
📁 Mounting Google Drive...
✅ Google Drive mounted
📥 Syncing checkpoints from shared Google Drive folder...
   Folder ID: 1EwJxSVjhLYD_2c1z2HDMCRRgyqux0tlv
   Downloading from: https://drive.google.com/drive/folders/1EwJxSVjhLYD_2c1z2HDMCRRgyqux0tlv
✅ Checkpoints successfully synced from shared folder!
✅ Checkpoint directory ready: /content/drive/MyDrive/checkpoints

...

Fold 1/5
======================================================================
Train chunks: 264, Val chunks: 66
Training batch-wise models (batch_size=5 chunks)...
  Batch 1/53 already trained, loading... ✓
  Batch 2/53 Training model on batch 2/53... ✓ (42.3s)
  Batch 3/53 already trained, loading... ✓
  ...
  Making ensemble predictions on validation (66 chunks)...
✅ Fold 1 - Loss: 0.1234, Accuracy: 0.9876
   Ensemble size: 53 models
   Saved 53 batch models for fold 1
```

## ✅ Чеклист перед запуском

- [ ] ID папки Google Drive обновлен в коде
- [ ] Папка доступна по ссылке
- [ ] Colab настроен на GPU (если доступно)
- [ ] Установлены все зависимости (`pip install -q kaggle gdown`)
- [ ] Google Drive смонтирован

---

**Готово!** Теперь вы можете обучать модель на разных Colab аккаунтах с автоматической синхронизацией! 🚀

























