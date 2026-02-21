import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Считываем train.csv, явно преобразуем столбец к str
df = pd.read_csv('train.csv', low_memory=False)
df['app_service'] = df['app_service'].astype(str)  # приведение к строке

label_encoder = LabelEncoder()
label_encoder.fit(df['app_service'].unique())

joblib.dump(label_encoder, 'label_encoder.pkl')
print("✅ label_encoder.pkl успешно создан!")
