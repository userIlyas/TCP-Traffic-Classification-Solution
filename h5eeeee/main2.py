import os
import joblib
import gc
import numpy as np
import pandas as pd

CHECKPOINT_DIR = './'
N_SPLITS = 5
TARGET_COL = 'app_service'
feature_cols_path = os.path.join(CHECKPOINT_DIR, 'train_feature_columns.pkl')

# ----- 0. Восстановление train_feature_columns.pkl при необходимости -----
if not os.path.exists(feature_cols_path):
    print('Автоматическое создание train_feature_columns.pkl ...')
    try:
        model = joblib.load(os.path.join(CHECKPOINT_DIR, 'sgd_fold_1.pkl'))
        feature_cols = list(model.feature_names_in_)
        joblib.dump(feature_cols, feature_cols_path)
        print(f'Успешно восстановлен список признаков, всего {len(feature_cols)}')
    except Exception as e:
        print('Не удалось восстановить feature columns из модели! Проверь версию sklearn или выгрузи вручную.')
        raise e
else:
    feature_cols = joblib.load(feature_cols_path)
    print('train_feature_columns.pkl найден. Используется {0} признаков.'.format(len(feature_cols)))

# ------ 1. Загрузка моделей и label_encoder ------
trained_models_path = os.path.join(CHECKPOINT_DIR, 'trained_models.pkl')
data = joblib.load(trained_models_path)
models = data['models']
label_encoder = data['label_encoder']

# ------ 2. Загружаем готовые test_chunks + test ids ------
chunk_index_path = os.path.join(CHECKPOINT_DIR, 'test_chunk_index.pkl')
if not os.path.exists(chunk_index_path):
    raise Exception("test_chunk_index.pkl не найден! Скопируй его/нужные pkl из Colab.")
chunk_index = joblib.load(chunk_index_path)
test_feature_files = chunk_index['feature_files']
test_ids = chunk_index['test_ids']
print(f"Найдено {len(test_feature_files)} chunk-файлов, всего тестовых id: {len(test_ids)}")

# ------ 3. Предсказания по чанкам ------
all_predictions = []
for i, feat_path in enumerate(test_feature_files):
    print(f"Predicting test chunk {i+1}/{len(test_feature_files)}...")
    X_chunk = joblib.load(feat_path)
    # Выравниваем структуру признаков под обучающую
    X_chunk = X_chunk.reindex(columns=feature_cols, fill_value=0)
    X_chunk = X_chunk.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_chunk_array = X_chunk.values.astype(np.float32)
    preds_list = [model.predict_proba(X_chunk_array) for model in models]
    chunk_preds = np.mean(preds_list, axis=0)
    all_predictions.append(chunk_preds)
    gc.collect()

# ------ 4. Финальная сборка submission ------
print("Combining predictions and creating submission...")
predictions = np.vstack(all_predictions)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)
submission = pd.DataFrame({'id': test_ids[:len(predicted_labels)], TARGET_COL: predicted_labels})
submission.to_csv('submission.csv', index=False)
print(f"✅ Submission saved: submission.csv, rows: {len(submission)}")
print("\nAll done! Pipeline completed locally.")

# --- Никакой пересоздании test, никакой обработки test.csv, только инференс по существующим chunk-файлам! ---
