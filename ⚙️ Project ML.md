

| **🟢 Basic (Новичок)** | ✅ Основы ML (что такое обучение модели)  <br>✅ Линейная и логистическая регрессия  <br>✅ Основные библиотеки (scikit-learn, TensorFlow) |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |

## 💼 Проект: **"AI-Рекрутер: Предсказание найма кандидата"**

### 🔍 Суть:

Создаем модель, которая предсказывает: **нанимают кандидата или нет** на основе данных из резюме (образование, опыт, навыки, soft skills и т.д.)

---

### 🧠 Ты освоишь:

| Тема | Как используется |
| --- | --- |
| ✅ **Основы ML** | Построение pipeline, обучение модели, метрики |
| ✅ **Линейная регрессия** | Прогноз зарплаты кандидата |
| ✅ **Логистическая регрессия** | Предсказание найма (да/нет) |
| ✅ **Scikit-learn** | Весь pipeline, препроцессинг, обучение, метрики |
| ✅ **TensorFlow** | Альтернативная реализация той же модели — для сравнения и GPU |
| ✅ **Работа с GPU** | Обучение модели в TensorFlow с ускорением |
| ✅ **Pandas, Matplotlib** | Чистка данных и визуализация |

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# 1. Генерация данных
np.random.seed(42)
n_samples = 1000

data = {
    "age": np.random.randint(20, 50, size=n_samples),
    "experience_years": np.random.randint(0, 15, size=n_samples),
    "education_level": np.random.choice(["High School", "Bachelor", "Master", "PhD"], size=n_samples),
    "knows_python": np.random.choice([0, 1], size=n_samples),
    "has_soft_skills": np.random.choice([0, 1], size=n_samples),
    "ml_course_completed": np.random.choice([0, 1], size=n_samples),
    "industry": np.random.choice(["Finance", "Tech", "Healthcare", "Education"], size=n_samples),
    "region": np.random.choice(["North", "South", "West", "East"], size=n_samples),
    "salary_expectation": np.random.randint(30000, 120000, size=n_samples),
}

prob_hired = (
    0.2 * data["experience_years"] +
    0.3 * data["knows_python"] +
    0.2 * data["ml_course_completed"] +
    0.2 * data["has_soft_skills"] +
    np.random.normal(0, 0.1, n_samples)
)
data["is_hired"] = (prob_hired > np.percentile(prob_hired, 60)).astype(int)

df = pd.DataFrame(data)

# 2. OneHot Encoding
df = pd.get_dummies(df, columns=["education_level", "industry", "region"])

# 3. Масштабирование
numeric_cols = ["age", "experience_years", "knows_python", "has_soft_skills", "ml_course_completed", "salary_expectation"]
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 4. Делим на признаки и метку
X = df.drop("is_hired", axis=1)
y = df["is_hired"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Обучение модели
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Предсказания и метрики
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

# 7. Визуализация метрик
metrics = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall
}
plt.figure(figsize=(6, 4))
plt.bar(metrics.keys(), metrics.values())
plt.title("Оценка модели логистической регрессии")
plt.ylim(0, 1)
plt.ylabel("Значение")
plt.grid(True)
plt.tight_layout()
plt.show()

(conf_matrix, accuracy, precision, recall)
```

Привет! Рад помочь 😊 Давай разберёмся с этими этапами, которые обычно встречаются в процессе построения модели машинного обучения:

---

### 🎲 **Генерация данных**

Это этап, на котором **собираются или создаются данные**, с которыми будет работать модель.

- В реальных задачах данные могут приходить из файлов, баз данных, API или сенсоров.
- В учебных целях часто **генерируются искусственные данные** (например, с помощью `sklearn.datasets.make_classification` или `make_regression`).
- Цель — получить **набор примеров**, состоящий из признаков (features) и целевой переменной (target), например:
    
    ```
    X = [[1.2, 3.4], [2.1, 1.1], ...]   # признаки
    y = [0, 1, 0, 1, ...]               # метки классов или значения
    
    ```
    

---

### 🧠 **Обработка признаков (Feature Engineering / Preprocessing)**

Перед обучением модель нужно **подготовить данные**:

- **Масштабирование признаков** (например, StandardScaler, MinMaxScaler)
- **Заполнение пропусков** (`fillna()` или `SimpleImputer`)
- **Кодирование категориальных признаков** (OneHotEncoding, LabelEncoding)
- **Удаление выбросов, нормализация, извлечение новых признаков**

Цель — привести данные в формат, который понятен модели и улучшает её работу.

---

### ⚙️ **Обучение модели**

Здесь происходит **тренировка модели на данных**:

- Делим данные на **обучающую и тестовую выборки** (`train_test_split`)
- Выбираем модель (например, `LogisticRegression`, `RandomForestClassifier`)
- Обучаем модель на тренировочных данных (`model.fit(X_train, y_train)`)
- Проверяем качество на отложенных данных (`model.predict(X_test)`)

---

### 📊 **Метрики**

Метрики нужны для **оценки качества модели**:

- Для классификации: `accuracy`, `precision`, `recall`, `f1-score`, `ROC-AUC`
- Для регрессии: `MAE`, `MSE`, `RMSE`, `R²`
- Пример:
    
    ```python
    from sklearn.metrics import accuracy_score
    accuracy_score(y_test, y_pred)
    
    ```
    

---

Хочешь пример на Python, где всё это показано пошагово?



| **🔵 Medium (Средний)** | ✅ Деревья решений, случайные леса (Decision Trees, Random Forest) <br>✅ KNN, SVM <br>✅ Кластеризация (K-Means, DBSCAN) |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- |
### ✅ **Проект 1: AI-Финансовый советник: Сегментация, прогноз и рекомендации для инвесторов**

**Что внутри:**

- **Кластеризация (K-Means, DBSCAN)**: Находишь кластеры среди клиентов по поведению (доходы, расходы, цели, риск-профиль).
- **KNN/SVM**: Предсказываешь принадлежность нового клиента к кластеру (портфель инвестора).
- **Random Forest/Decision Tree**: Предсказываешь, какой продукт предложить клиенту (облигации, акции, крипта и т.д.).
- **Визуализация данных и кластеров в 2D/3D**: Визуализируешь результаты кластеризации.

**Почему это интересно:**

- Очень приближено к реальной индустрии (финансы + ML), можно использовать открытые финансовые датасеты, добавить генерацию фейковых пользователей.

```python

# ✅ Проект: AI-Финансовый советник: Сегментация, прогноз и рекомендации для инвесторов

# 🔧 Импорты
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import networkx as nx

# 📊 Генерация синтетических данных
np.random.seed(42)
n_clients = 100_000

ages = np.random.randint(18, 65, size=n_clients)
incomes = np.random.randint(150_000, 1_500_000, size=n_clients)
expenses = np.random.randint(50_000, 800_000, size=n_clients)
risk_profiles = np.random.choice(["Низкий", "Средний", "Высокий"], size=n_clients, p=[0.4, 0.4, 0.2])
investment_goals = np.random.choice(["Стабильность", "Рост капитала", "Пассивный доход"], size=n_clients)

df = pd.DataFrame({
    "age": ages,
    "income": incomes,
    "expenses": expenses,
    "risk_profile": risk_profiles,
    "investment_goal": investment_goals
})

# 💡 Генерация рекомендаций
df['recommendation'] = df['risk_profile'].map({
    "Высокий": "Криптовалюта",
    "Средний": "Акции",
    "Низкий": "Облигации"
})

# 🗃️ Кодирование категориальных признаков
encoded_df = df.copy()
le_risk = LabelEncoder()
le_goal = LabelEncoder()
encoded_df['risk_profile_enc'] = le_risk.fit_transform(encoded_df['risk_profile'])
encoded_df['investment_goal_enc'] = le_goal.fit_transform(encoded_df['investment_goal'])

# ♻️ Кластеризация
features = encoded_df[['age', 'income', 'expenses', 'risk_profile_enc', 'investment_goal_enc']]
kmeans = KMeans(n_clusters=3, random_state=42)
encoded_df['cluster'] = kmeans.fit_predict(features)

# 🧬 Анализ кластеров
cluster_summary = encoded_df.groupby('cluster').agg({
    'age': ['mean', 'min', 'max'],
    'income': ['mean', 'min', 'max'],
    'expenses': ['mean', 'min', 'max'],
    'risk_profile': lambda x: x.value_counts().to_dict(),
    'investment_goal': lambda x: x.value_counts().to_dict()
})
cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
cluster_summary.reset_index(inplace=True)
print("\\n📊 Анализ кластеров:")
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

print(cluster_summary)

# 🪪 Создание возрастных групп
bins = [18, 25, 35, 45, 55, 65]
labels = ["18-25", "26-35", "36-45", "46-55", "56-65"]
encoded_df['age_group'] = pd.cut(encoded_df['age'], bins=bins, labels=labels, right=False)

# 💰 Финансовый остаток
encoded_df['portfolio_balance'] = encoded_df['income'] - encoded_df['expenses']

# 🧩 Группировка по возрастным группам
age_group_summary = encoded_df.groupby('age_group').agg(
    clients=('age', 'count'),
    avg_income=('income', 'mean'),
    avg_expenses=('expenses', 'mean'),
    avg_balance=('portfolio_balance', 'mean'),
    total_balance=('portfolio_balance', 'sum')
).reset_index()
print("\\n📊 Финансовая сводка по возрастам:")
print(age_group_summary)

# 📊 Построение графика
plt.figure(figsize=(12, 6))
bar_width = 0.25
x = np.arange(len(age_group_summary['age_group']))

plt.bar(x - bar_width, age_group_summary['avg_income'], width=bar_width, label='Средний доход')
plt.bar(x, age_group_summary['avg_expenses'], width=bar_width, label='Средние расходы')
plt.bar(x + bar_width, age_group_summary['avg_balance'], width=bar_width, label='Средний остаток')

plt.xticks(x, age_group_summary['age_group'])
plt.title("📊 Финансовое сравнение по возрастным группам")
plt.xlabel("Возрастная группа")
plt.ylabel("Сумма (₸)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🤖 Обучение модели
X = encoded_df[['age', 'income', 'expenses', 'risk_profile_enc', 'investment_goal_enc']]
y = encoded_df['recommendation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 📈 Метрики
print("\\n📈 Классификация рекомендаций:")
print(classification_report(y_test, y_pred))

# 🪢 Уникальные узлы
age_groups = list(encoded_df['age_group'].astype(str).unique())
risk_profiles = ["Низкий", "Средний", "Высокий"]
recommendations = ["Облигации", "Акции", "Криптовалюта"]

# Создаём граф
G = nx.DiGraph()

# Группировка для расчёта количества
from collections import Counter

sample_df = encoded_df.sample(5000, random_state=42)
group_to_risk = Counter()
risk_to_recommendation = Counter()

for _, row in sample_df.iterrows():
    group = str(row['age_group'])
    risk = row['risk_profile']
    rec = row['recommendation']
    group_to_risk[(group, risk)] += 1
    risk_to_recommendation[(risk, rec)] += 1

# Добавляем связи
edge_colors = []
edge_widths = []

for (group, risk), weight in group_to_risk.items():
    G.add_edge(group, risk, weight=weight)
    color = {"Низкий": "green", "Средний": "orange", "Высокий": "red"}[risk]
    edge_colors.append(color)
    edge_widths.append(weight / 150)

for (risk, rec), weight in risk_to_recommendation.items():
    G.add_edge(risk, rec, weight=weight)
    color = {"Низкий": "green", "Средний": "orange", "Высокий": "red"}[risk]
    edge_colors.append(color)
    edge_widths.append(weight / 150)

# Ручное позиционирование узлов
pos = {}
y_age = 2
y_risk = 1
y_rec = 0

for i, group in enumerate(sorted(age_groups)):
    pos[group] = (i, y_age)

for i, risk in enumerate(risk_profiles):
    pos[risk] = (i + 1, y_risk)

for i, rec in enumerate(recommendations):
    pos[rec] = (i + 1, y_rec)

# Рисуем
plt.figure(figsize=(14, 8))
nx.draw(G, pos, with_labels=True, edge_color=edge_colors, width=edge_widths,
        node_size=2500, node_color="lightblue", font_size=10)

plt.title("📊 Чёткая схема: Возраст → Риск → Рекомендация", fontsize=14)
plt.axis("off")
plt.text(2, 2.5,  # 📍 x=7, y=2.5 — можно подстроить
    "🎨 Расшифровка цвета стрелок:\\n"
    "🟥 Красный   → Высокий риск (Клиенты готовы рисковать) → 💡Рекомендация: Криптовалюта\\n"
    "🟧 Оранжевый → Средний риск (Сбалансированное отношение к риску) → 💼 Рекомендация: Акции\\n"
    "🟩 Зелёный   → Низкий риск (Осторожные, предпочитают стабильность)  → 🏦 Рекомендация: Облигации",
    fontsize=10,
    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
)
plt.tight_layout()
plt.show()

```

### ✅ **Проект 2: AI-Служба Поддержки: Классификация тикетов и автоматическая маршрутизация** 💬

📦 **Что внутри:**

- **NLP** — предобработка текста тикетов (TF-IDF или BERT-эмбеддинги).
- **Кластеризация тикетов (K-Means/DBSCAN)** — группировка по теме (например, проблемы с продуктом, технические запросы).
- **Decision Tree/Random Forest** — классификация тикетов и автоматическая маршрутизация к соответствующим отделам.

📌 **Почему это интересно:**

- Автоматизация обработки тикетов для повышения скорости и качества обслуживания в службе поддержки.

```python
# 🧠 NLP-проект: "AI-Служба Поддержки"
# Цель: классифицировать тикеты (обращения клиентов) по отделам:
# → Техподдержка, Финансы, Удаление аккаунта, Жалобы, Общие вопросы

# 🔧 Импорты

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ===============================
# 🧠 Шаг 1: Датасет
# ===============================

categories = {
    "Техподдержка": [
        "Не работает вход в приложение",
        "Сбой при запуске",
        "Выдает ошибку при загрузке"
    ],
    "Финансы": [
        "Оплата прошла дважды",
        "Хочу вернуть деньги",
        "Списали деньги без моего согласия"
    ],
    "Удаление": [
        "Прошу удалить мой аккаунт",
        "Удалите мои данные",
        "Хочу полностью выйти из системы"
    ],
    "Жалобы": [
        "Хочу пожаловаться на оператора",
        "Неправильное обращение с клиентом",
        "Не получил обещанный бонус"
    ],
    "Общие": [
        "Как получить промокод?",
        "Где найти инструкцию?",
        "Есть ли мобильная версия?"
    ]
}

data = {
    "ticket": [],
    "label": []
}

for label, phrases in categories.items():
    for phrase in phrases:
        data["ticket"].append(phrase)
        data["label"].append(label)

df = pd.DataFrame(data)

df = pd.DataFrame(data)

# ===============================
# 🔠 Шаг 2: TF-IDF векторизация
# ===============================

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['ticket'])
y = df['label']

# ===============================
# 🔁 Шаг 3: Кластеризация (KMeans)
# ===============================

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)
df['cluster'] = clusters
print("\\n📊 Кластеризация (без меток):")
print(df[['ticket', 'cluster']])

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# PCA — для уменьшения размерности, иначе DBSCAN не справится с TF-IDF
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X.toarray())

# DBSCAN кластеризация
dbscan = DBSCAN(eps=0.5, min_samples=2)
clusters_db = dbscan.fit_predict(X_reduced)

df['dbscan_cluster'] = clusters_db
print(df[['ticket', 'dbscan_cluster']])

# ===============================
# 🌳 Шаг 4: Random Forest
# ===============================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)

y_pred_rf = clf_rf.predict(X_test)
print("\\n🧪 Random Forest:")
print(classification_report(y_test, y_pred_rf))

# ===============================
# 🌲 Шаг 5: Decision Tree
# ===============================

clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)

y_pred_tree = clf_tree.predict(X_test)
print("\\n🧪 Decision Tree:")
print(classification_report(y_test, y_pred_tree))

# ===============================
# 📉 Шаг 6: Confusion Matrix
# ===============================

ConfusionMatrixDisplay.from_estimator(clf_rf, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix (Random Forest)")
plt.show()

# ===============================
# 🤖 Шаг 7: BERT-вариант (если хочешь PRO-версию)
# ===============================

from sentence_transformers import SentenceTransformer

model_bert = SentenceTransformer('all-MiniLM-L6-v2')
X_bert = model_bert.encode(df['ticket'].tolist())

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_bert, y,test_size=0.3, random_state=42)
clf_bert = RandomForestClassifier()
clf_bert.fit(X_train_b, y_train_b)

y_pred_bert = clf_bert.predict(X_test_b)
print("\\n🤖 BERT + Random Forest:")
print(classification_report(y_test_b, y_pred_bert))
```

### ✅ **Проект 3: AI-Диагностика заболеваний с помощью медицинских изображений** 🏥

📦 **Что внутри:**

- **Convolutional Neural Networks (CNN)** — обучаем модель распознавать заболевания на медицинских изображениях (например, рентген или МРТ).
- **Data Augmentation** — используем аугментацию данных для увеличения объёма обучающих данных.
- **Transfer Learning** — применяем уже обученные модели (например, VGG16, ResNet) для улучшения результатов.

📌 **Почему это интересно:**

- Применение в медицине для диагностики заболеваний на ранних стадиях, что может существенно улучшить качество жизни пациентов и ускорить лечение.

```python

# 🔬 Проект 3: AI-Медицинская диагностика
# 📍Цель: Обнаружение заболеваний по изображениям
# 📦 Что будет внутри:

# 📂 Загрузка медицинских изображений (рентген / МРТ)
# 🧠 Обучение CNN (Convolutional Neural Network)
# ⚙️ Использование Transfer Learning (VGG16 / ResNet)
# 🔄 Data Augmentation (увеличение обучающей выборки)
# 📈 Метрики: точность, матрица ошибок
# 🧭 Визуализация предсказаний

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

# Пути к изображениям
base_dir = r"C:\\Users\\Madina\\OneDrive\\Документы\\Python Scripts\\train"
normal_dir = os.path.join(base_dir, "NORMAL")
pneumonia_dir = os.path.join(base_dir, "PNEUMONIA")

# Загружаем по 3 изображения
normal_images = glob.glob(os.path.join(normal_dir, "*.jpeg"))[:3]
pneumonia_images = glob.glob(os.path.join(pneumonia_dir, "*.jpeg"))[:3]

# Создаём подграфики
fig, axs = plt.subplots(2, 3, figsize=(12, 6))

for i, img_path in enumerate(normal_images):
    img = mpimg.imread(img_path)
    axs[0, i].imshow(img, cmap='gray')
    axs[0, i].set_title("NORMAL")
    axs[0, i].axis('off')

for i, img_path in enumerate(pneumonia_images):
    img = mpimg.imread(img_path)
    axs[1, i].imshow(img, cmap='Blues_r')
    axs[1, i].set_title("PNEUMONIA")
    axs[1, i].axis('on')

plt.suptitle("🫁 Chest X-Ray: NORMAL vs PNEUMONIA", fontsize=16)
plt.tight_layout()
plt.show()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 📁 Путь к твоей тренировочной папке
train_dir = r"C:\\Users\\Madina\\OneDrive\\Документы\\Python Scripts\\train"

# ⚙️ Генератор с масштабированием и аугментацией (можно без неё для начала)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# 📤 Генератор для обучения
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),   # Все изображения будут 150x150 пикселей
    batch_size=32, 
    class_mode='binary',  # У нас два класса: 0 (NORMAL) и 1 (PNEUMONIA)
    subset='training',
    color_mode='grayscale'  
)

# 📤 Генератор для валидации
val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32, 
    class_mode='binary',  # У нас два класса: 0 (NORMAL) и 1 (PNEUMONIA)
    subset='validation',
    color_mode='grayscale'  
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 🎯 Простая CNN-модель
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)), # 1 — так как grayscale
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'), 
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Для бинарной классификации
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Обучение модели
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Визуализировать Accuracy и Loss по эпохам:
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("📈 Accuracy по эпохам")
plt.xlabel("Эпоха")
plt.ylabel("Точность")
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("📉 Потери по эпохам")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# ✅ Состояние проекта: AI-диагностика по рентгену

# Импорты
import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# === 🔧 Функции ===

from PIL import Image

def avg_brightness(image_path):
    try:
        img = Image.open(image_path).convert('L')  # 'L' = grayscale
        return np.array(img).mean()
    except Exception as e:
        print(f"⚠️ Ошибка при чтении {image_path}: {e}")
        return 0

def get_feature_vector(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()

# === 🧠 Модель VGG16 ===
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# === 📁 Проверка путей ===
for category in ['NORMAL', 'PNEUMONIA']:
    folder_path = os.path.join(r"C:\\Users\\Madina\\OneDrive\\Документы\\Python Scripts\\train", category)
    if not os.path.exists(folder_path):
        print(f"❌ Папка не найдена: {folder_path}")
        continue
    paths = glob.glob(os.path.join(folder_path, "*.jpeg"))
    print(f"✅ Найдено {len(paths)} файлов в {category}")

# === 📊 Сбор данных ===
data = []

for category in ['NORMAL', 'PNEUMONIA']:
    folder_path = os.path.join(r"C:\\Users\\Madina\\OneDrive\\Документы\\Python Scripts\\train", category)
    paths = glob.glob(os.path.join(folder_path, "*.jpeg"))
    sample_size = min(100, len(paths))
    sample_paths = random.sample(paths, sample_size)

    for path in sample_paths:
        brightness = avg_brightness(path)
        features = get_feature_vector(path)
        feature_sum = features.sum()
        data.append({
            "category": category,
            "path": path,
            "brightness": brightness,
            "feature_sum": feature_sum
        })

# === 📈 Визуализация ===
df = pd.DataFrame(data)
anomalies = df[(df['brightness'] < 50) | (df['brightness'] > 200)]
print(anomalies[['category', 'path', 'brightness']])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="brightness", y="feature_sum", hue="category", alpha=0.7)
plt.title("Анализ снимков: Яркость vs Признаки VGG16")
plt.xlabel("Средняя яркость")
plt.ylabel("Сумма признаков (VGG16)")
plt.grid(True)
plt.tight_layout()
plt.show()

import plotly.express as px

df["filename"] = df["path"].apply(os.path.basename)

fig = px.scatter(
    df,
    x="brightness",
    y="feature_sum",
    color="category",
    hover_data=["filename", "category", "brightness", "feature_sum"],
    title="🧠 Интерактивный анализ снимков: Яркость vs Признаки VGG16"
)

fig.show()

```