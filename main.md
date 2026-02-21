# Классификация приложений по зашифрованным TCP-потокам

В этой тетради разбираем, как по последовательности длин TCP-пакетов определить приложение-источник (`app_service`). Полезная нагрузка зашифрована, поэтому единственное, что мы видим, — размеры и направления пакетов (положительное значение — отправка клиентом, отрицательное — получение). Мы подробно объясним каждую математическую операцию: что делаем, какая формула и почему это работает.

**Задача в терминах математики.** Пусть есть выборка
\(\mathcal{D} = \{(S^{(i)}, y^{(i)})\}_{i=1}^N\), где для каждого сеанса \(i\):
- \(S^{(i)} = (s^{(i)}_1, s^{(i)}_2, \ldots, s^{(i)}_{30}) \in \mathbb{Z}^{30}\) — вектор из 30 элементов (длины пакетов с учётом знака).
- \(s^{(i)}_j > 0\) — пакет ушёл от клиента; \(s^{(i)}_j < 0\) — пакет получен; нули — добивка, если реальный поток короче 30 пакетов.
- \(y^{(i)} \in \{c_1, c_2, \ldots, c_K\}\) — класс/приложение. Нужно построить модель \(f: \mathbb{R}^{30} \to \{c_1,\ldots,c_K\}\), которая по \(S^{(i)}\) предскажет \(y^{(i)}\).

Ниже каждый шаг подготовки данных и обучения сопровождается текстовой ячейкой с формулами и интуицией. Так ноутбук остаётся исполнимым, но понятным без глубоких знаний математики.

```python
# 0. pip install
%pip install -q pandas==2.2.2 numpy==1.26.4 scikit-learn==1.5.2 lightgbm==4.5.0 tqdm kaggle==1.6.9

import os, json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def setup_kaggle():
    kaggle_dir = os.path.join(os.getcwd(), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    kaggle_credentials = {
        "username": "YOUR_KAGGLE_USERNAME",
        "key": "YOUR_KAGGLE_API_KEY"
    }
    cred_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(cred_path, "w") as f:
        json.dump(kaggle_credentials, f)
    os.chmod(cred_path, 0o600)
    os.environ["KAGGLE_CONFIG_DIR"] = kaggle_dir
    print("✅ Kaggle API configured")
    if not os.path.exists("train.csv"):
        print("📥 Downloading competition data...")
        os.system("kaggle competitions download -c whos-talking-classify-the-app-by-its-packets -p .")
        os.system("unzip -q whos-talking-classify-the-app-by-its-packets.zip")
        print("✅ Data downloaded and extracted")
    else:
        print("✅ Data already exists")

setup_kaggle()

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from lightgbm import LGBMClassifier

TARGET_COL = "app_service"
TCP_COLS = [f"tcp_len_{i}" for i in range(1, 31)]

# ======== 1. Загрузка данных ========
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
```

## Шаг 0–1. Окружение и чтение данных
**Что делаем?** Ставим зависимости, настраиваем Kaggle API, считываем `train.csv` и `test.csv`, определяем, какие столбцы являются последовательностью пакетов.

**Математическая формулировка.** После чтения получаем две матрицы: 
- \(X_{\text{train}} \in \mathbb{Z}^{N \times 30}\) — строки \(S^{(i)}\) для обучающих сессий.
- \(X_{\text{test}} \in \mathbb{Z}^{M \times 30}\) — строки для тестовых сессий (без меток).
Целевой вектор \(y \in \{c_1,\ldots,c_K\}^N\) берётся из столбца `app_service`.

**Почему это работает?** Мы чётко фиксируем, что каждый столбец `tcp_len_j` — это позиция \(j\) в векторе \(S^{(i)}\). Зная форму матриц (кол-во строк и 30 столбцов), мы можем далее формулировать операции как преобразования над матрицами/векторами без обращения к содержимому пакетов.

## Шаг 2. Слияние крупных подряд идущих пакетов
**Что делаем?** Превращаем исходный вектор пакетов \(S^{(i)} = (s^{(i)}_1,\ldots,s^{(i)}_{30})\) в новый вектор \(\tilde S^{(i)}\), где подряд идущие большие пакеты одного направления объединяются в один суммарный пакет. Это попытка восстановить «смысловые блоки» передачи данных, скрытые фрагментацией TCP.

**Математическая формула.** Пусть \(B = 1200\) — порог «крупного» пакета. Для последовательности \(s^{(i)}_j\):
- Определяем знак \(\text{sign}(s^{(i)}_j) \in \{-1, +1\}\).
- Если \(|s^{(i)}_j| > B\) и \(|s^{(i)}_{j-1}| > B\) и знаки совпадают, то объединяем: \(\tilde s^{(i)}_{k} = \tilde s^{(i)}_{k} + s^{(i)}_j\).
- Иначе начинаем новый элемент \(\tilde s^{(i)}_{k+1} = s^{(i)}_j\).
- После обработки реальных пакетов добиваем вектор нулями до длины 30: \(\tilde s^{(i)}_t = 0\) для пустых позиций.

**Почему это работает?** В реальности большой файл дробится на сегменты размером близким к MTU (~1500 байт). Последовательность одинаковых по направлению крупных пакетов часто представляет один логический блок данных. Складывая их, мы:
- уменьшаем «шум» от сетевой фрагментации;
- приближаем наблюдаемую последовательность к исходному размеру переданного блока;
- сохраняем знак, поэтому направление (клиент→сервер или сервер→клиент) не теряется.

**Пример на матрице.** Если строка матрицы \(X_{\text{train}}\) выглядит как \((+1300, +1250, +300, -1400, -1300, 0, 0, \ldots)\), то после слияния получим \((+2550, +300, -2700, 0, 0, \ldots)\). Первые два положительных >1200 сложились, два отрицательных >1200 сложились, остальное осталось. Такой вектор \(\tilde S\) идёт дальше в признаки.

```python
# ======== 2. Слияние пакетов =========
def merge_packets_row(values: np.ndarray) -> np.ndarray:
    merged = []
    last_sign = None
    last_big = False
    for v in values:
        if v == 0:
            break
        sign = 1 if v > 0 else -1
        big = abs(v) > 1200
        if big and last_big and sign == last_sign and merged:
            merged[-1] += v
        else:
            merged.append(v)
        last_sign = sign
        last_big = big
    merged += [0] * (30 - len(merged))
    return np.array(merged)

train_vals = train_df[TCP_COLS].values
test_vals = test_df[TCP_COLS].values

train_merged = np.array([merge_packets_row(x) for x in tqdm(train_vals, desc="Merging train")])
test_merged = np.array([merge_packets_row(x) for x in tqdm(test_vals, desc="Merging test")])

train_df[TCP_COLS] = train_merged
test_df[TCP_COLS] = test_merged
```

## Шаг 3. Построение признаков из последовательности пакетов
**Что делаем?** По каждой строке матрицы пакетов \(\tilde S^{(i)}\) считаем сводные статистики: количество пакетов, суммарные байты, разбивки на «малые/средние/крупные», смены направления и т.д. Это превращает сырую последовательность длиной 30 в вектор признаков \(\phi(\tilde S^{(i)})\) фиксированной длины.

**Математические формулы (для строки \(\tilde S = (\tilde s_1,\ldots,\tilde s_{30})\)):**
- `total_packets`: \(\sum_j \mathbf{1}[\tilde s_j \neq 0]\).
- `total_bytes`: \(\sum_j |\tilde s_j|\).
- `mean_packet_size`: \(\frac{1}{30} \sum_j |\tilde s_j + \varepsilon|\), \(\varepsilon = 10^{-10}\) — микро-сдвиг, чтобы среднее не обнулялось, если вся строка равна нулю.
- `std_packet_size`: стандартное отклонение \(\sqrt{\frac{1}{30}\sum_j (\tilde s_j - \bar{s})^2}\).
- `max_packet_size`: \(\max_j |\tilde s_j|\).
- `min_packet_size`: \(\min_j |\tilde s_j + C|\), \(C = 10^{10}\) — все элементы сдвигаются на одинаковую константу. Значения становятся \(C \pm |\tilde s_j|\), поэтому различия между сессиями сохраняются в малой части \(|\tilde s_j|\), а константа делает признак строго положительным даже при нулевой строке.
- `outgoing_count` / `incoming_count`: \(\sum_j \mathbf{1}[\tilde s_j > 0]\) и \(\sum_j \mathbf{1}[\tilde s_j < 0]\).
- `outgoing_bytes` / `incoming_bytes`: \(\sum_j \max(\tilde s_j, 0)\) и \(\sum_j |\min(\tilde s_j, 0)|\).
- `traffic_ratio`: \(\frac{\text{outgoing\_bytes}}{\text{incoming\_bytes} + 1}\); «+1» защищает от деления на ноль, если входящего трафика нет.
- `large_packets`: \(\sum_j \mathbf{1}[|\tilde s_j| > 1200]\).
- `small_packets`: \(\sum_j \mathbf{1}[0 < |\tilde s_j| \le 100]\).
- `medium_packets`: \(\sum_j \mathbf{1}[100 < |\tilde s_j| \le 1200]\).
- `direction_changes`: \(\sum_{j=2}^{30} \mathbf{1}[\text{sign}(\tilde s_j) \neq \text{sign}(\tilde s_{j-1})]\) — сколько раз поток менял направление.
- `burst_count`: \(\sum_{j=2}^{30} \mathbf{1}[|\tilde s_j - \tilde s_{j-1}| > 500]\) — резкие скачки между соседними пакетами.
- `first_packet`, `second_packet`, `third_packet`: \(|\tilde s_1|, |\tilde s_2|, |\tilde s_3|\).
- `first_direction`: \(\text{sign}(\tilde s_1)\).
- `q25`, `q50`, `q75`: процентили 25/50/75 распределения \(|\tilde s_j|\).
- `iqr`: интерквартильный размах \(q_{75} - q_{25}\).
- `entropy`: строим гистограмму \(|\tilde s_j|\) на 10 бинов, получаем вероятности \(p_b\), считаем \(-\sum_b p_b \log_2(p_b + \varepsilon)\). Чем разнообразнее размеры пакетов, тем выше энтропия.
- `cv` (коэффициент вариации): \(\frac{\text{std}}{\text{mean} + \varepsilon}\) — относительная изменчивость.

**Как это работает на примере матрицы.** Пусть есть строка \((+1500, -800, 0, \ldots)\). Тогда:
- `total_packets = 2`, `total_bytes = 2300`.
- `mean_packet_size \approx (1500 + 800 + 28\cdot \varepsilon)/30` — нули внесли только микроскопический вклад.
- `min_packet_size = \min(|1500 + C|, |{-800} + C|, |0 + C|,\ldots) \approx C - 800` — все значения подняты к \(C\), но различия зависят от реального минимума; константа одинакова для всех строк, поэтому деревья сравнивают именно добавки от \(|\tilde s_j|\).
Эти признаки переводят последовательность в компактный, но информативный числовой портрет потока.

```python
# ======== 3. Feature Engineering (продвинутый набор фичей) ========
def create_network_features(df):
    data = df[TCP_COLS].values
    features = {}
    features['total_packets'] = np.count_nonzero(data, axis=1)
    features['total_bytes'] = np.sum(np.abs(data), axis=1)
    features['mean_packet_size'] = np.mean(np.abs(data + 1e-10), axis=1)
    features['std_packet_size'] = np.std(data, axis=1)
    features['max_packet_size'] = np.max(np.abs(data), axis=1)
    features['min_packet_size'] = np.min(np.abs(data + 1e10), axis=1)
    features['outgoing_count'] = np.sum(data > 0, axis=1)
    features['incoming_count'] = np.sum(data < 0, axis=1)
    features['outgoing_bytes'] = np.sum(np.where(data > 0, data, 0), axis=1)
    features['incoming_bytes'] = np.sum(np.abs(np.where(data < 0, data, 0)), axis=1)
    features['traffic_ratio'] = features['outgoing_bytes'] / (features['incoming_bytes'] + 1)
    features['large_packets'] = np.sum(np.abs(data) > 1200, axis=1)
    features['small_packets'] = np.sum((np.abs(data) > 0) & (np.abs(data) <= 100), axis=1)
    features['medium_packets'] = np.sum((np.abs(data) > 100) & (np.abs(data) <= 1200), axis=1)
    features['direction_changes'] = np.sum(np.diff(np.sign(data), axis=1) != 0, axis=1)
    features['burst_count'] = np.sum(np.abs(np.diff(data, axis=1)) > 500, axis=1)
    features['first_packet'] = np.abs(data[:, 0])
    features['second_packet'] = np.abs(data[:, 1])
    features['third_packet'] = np.abs(data[:, 2])
    features['first_direction'] = np.sign(data[:, 0])
    features['q25'] = np.percentile(np.abs(data), 25, axis=1)
    features['q50'] = np.percentile(np.abs(data), 50, axis=1)
    features['q75'] = np.percentile(np.abs(data), 75, axis=1)
    features['iqr'] = features['q75'] - features['q25']
    def calc_entropy(row):
        nonzero = row[row != 0]
        if len(nonzero) == 0:
            return 0
        bins = np.histogram(np.abs(nonzero), bins=10)[0]
        probs = bins / (bins.sum() + 1e-10)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs + 1e-10))
    features['entropy'] = np.apply_along_axis(calc_entropy, 1, data)
    features['cv'] = features['std_packet_size'] / (features['mean_packet_size'] + 1e-10)
    return pd.DataFrame(features)

print("Creating features...")
train_features = create_network_features(train_df)
test_features = create_network_features(test_df)
```

## Шаг 4. Объединение исходных значений и новых признаков
**Что делаем?** Склеиваем исходные 30 позиций \(\tilde S^{(i)}\) и вычисленные признаки \(\phi(\tilde S^{(i)})\) в один большой вектор. Это даёт модели и «сырые» размеры пакетов, и агрегаты.

**Математическая формула.** Для каждой сессии формируем 
\[ X^{(i)} = [\tilde S^{(i)} \,\, \phi(\tilde S^{(i)})] \in \mathbb{R}^d, \]
где \(d = 30 + \text{количество новых признаков}\).

**Почему это работает?** Решающее дерево (и бустинг) может использовать как отдельные элементы последовательности (например, первый пакет), так и сводные показатели (энтропия, количество смен направления). Комбинация повышает информативность без изменения исходных данных.

```python
X_train_full = pd.concat([train_df[TCP_COLS].reset_index(drop=True),
                          train_features.reset_index(drop=True)], axis=1)
X_test_full = pd.concat([test_df[TCP_COLS].reset_index(drop=True),
                         test_features.reset_index(drop=True)], axis=1)
```

## Шаг 5. Кодирование целевой переменной и чистка матрицы признаков
**Что делаем?** Превращаем текстовые метки приложений `app_service` в целые числа (для мультиклассовой функции потерь) и убираем бесконечные/NaN значения из матриц признаков.

**Математические формулы.**
- `LabelEncoder` строит биекцию \(g: \{c_1,\ldots,c_K\} \to \{0,1,\ldots,K-1\}\). Получаем \(y^{(i)}_{\text{int}} = g(y^{(i)})\).
- Чистка признаков: заменяем \(+\infty, -\infty\) на `NaN`, затем `NaN` на 0. Итоговую матрицу назовём \(\hat X\).

**Почему это работает?**
- Многоклассовый бустинг в LightGBM ожидает целочисленные метки классов.
- Бесконечности могут возникать при делении (например, если входящий трафик нулевой и `outgoing_bytes` огромен). Замена на 0 предотвращает числовые аварии в модели; деревья сами решат, полезен ли такой «нулевой» индикатор.

```python
le = LabelEncoder()
y = le.fit_transform(train_df[TARGET_COL].astype(str))
num_classes = len(np.unique(y))
print(f"Number of classes: {num_classes}")

X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan).fillna(0)
X_test_full = X_test_full.replace([np.inf, -np.inf], np.nan).fillna(0)
```

## Шаг 6. Обучение модели LightGBM
**Что делаем?** Настраиваем многоклассовый бустинг на решающих деревьях и обучаем на матрице признаков \(\hat X\) и метках \(y_{\text{int}}\).

**Математическая формула.** LightGBM минимизирует мультиклассовую логарифмическую потерю:
\[ L = - \sum_{i=1}^N \sum_{k=1}^K \mathbf{1}[y^{(i)} = k] \log p_k^{(i)}, \]
где \(p_k^{(i)}\) — вероятность класса \(k\) из ансамбля деревьев. Ансамбль строится итеративно: \(F_{t}(x) = F_{t-1}(x) + \eta \cdot h_t(x)\), где \(h_t\) — новое дерево, \(\eta = \text{learning\_rate}\).

**Почему это работает?**
- Деревья хорошо ловят нелинейности и взаимодействия признаков (например, сочетание «много входящего трафика» и «высокая энтропия»).
- Параметры (`max_depth`, `num_leaves`, `feature_fraction`, `bagging_fraction`) контролируют сложность и предотвращают переобучение, сохраняя скорость обучения.

```python
# ======== 4. Быстрая LightGBM-модель на всей выборке ========
lgb_params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.03,
    "num_leaves": 100,
    "max_depth": 10,  # Ограничить глубину для ускорения!
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": -1,
    "random_state": 42,
    "device_type": "cpu",
    "num_threads": 32,
    "n_estimators": 300,  # Быстрое обучение, early stopping отключен
}

model = LGBMClassifier(**lgb_params)
model.fit(X_train_full, y)
```

## Шаг 7. Предсказание на тесте и сохранение
**Что делаем?** Получаем вероятности классов на тестовой матрице \(X_{\text{test}}\), берём наиболее вероятный класс для каждой строки, переводим обратно из индексов в исходные названия приложений и сохраняем `submit.csv`.

**Математическая формула.** Для каждой тестовой строки \(x\) модель даёт вектор \(p(x) = (p_1,\ldots,p_K)\). Мы выбираем \(\hat{k} = \arg\max_k p_k\) и затем применяем обратное отображение \(g^{-1}(\hat{k})\), чтобы вернуть текстовую метку.

**Почему это работает?** \(\arg\max\) выбирает класс с наибольшей апостериорной вероятностью по модели. Файл `submit.csv` содержит пары (`id`, `app_service`), готовые для отправки на Kaggle.

```python
# ======== 5. Предсказание и сохранение ========
test_pred_int = model.predict(X_test_full)
test_df[TARGET_COL] = le.inverse_transform(test_pred_int)
test_df[["id", TARGET_COL]].to_csv("submit.csv", index=False)
print("\n✅ Saved submit.csv")
print(f"Submission shape: {test_df[['id', TARGET_COL]].shape}")
```



