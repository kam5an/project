import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow import keras

import warnings
warnings.filterwarnings('ignore')

# ── 1. ЗАГРУЗКА ДАННЫХ
print("=" * 55)
print("ШАГ 1: Загрузка данных")
print("=" * 55)

df = pd.read_csv(r'C:\\Users\\A003\\Desktop\\diss\\diss_practise\\Sample - Superstore.csv', encoding='latin-1')
print(f"Загружено строк: {len(df)}")
print(f"Столбцы: {list(df.columns)}\n")

# ── 2. ПРЕДОБРАБОТКА
print("=" * 55)
print("ШАГ 2: Предобработка данных")
print("=" * 55)

features = ['Category', 'Sub-Category', 'Region',
            'Segment', 'Ship Mode', 'Discount', 'Quantity']
target = 'Sales'
X = df[features].copy()
y = np.log1p(df[target])  # логарифмическое преобразование
# One-Hot Encoding категориальных переменных
cat_cols = ['Category', 'Sub-Category', 'Region', 'Segment', 'Ship Mode']
X = pd.get_dummies(X, columns=cat_cols)
# Стандартизация числовых переменных
num_cols = ['Discount', 'Quantity']
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
# Разбиение на обучающую (80%) и тестовую (20%) выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Обучающая выборка: {X_train.shape[0]} записей")
print(f"Тестовая выборка:  {X_test.shape[0]} записей")
print(f"Число признаков:   {X_train.shape[1]}\n")

# ── 3. ВСПОМОГАТЕЛЬНАЯ ФУНКЦИЯ ОЦЕНКИ
def evaluate(name, y_true_log, y_pred_log):
    """Переводим прогнозы обратно в доллары и считаем метрики."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  MAE:  ${mae:,.2f}")
    return rmse, mae, y_true, y_pred

results = {}

# ── 4. МОДЕЛЬ 1: ЛИНЕЙНАЯ РЕГРЕССИЯ
print("=" * 55)
print("ШАГ 3: Линейная регрессия (baseline)")
print("=" * 55)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

rmse, mae, y_true, y_pred_lr = evaluate("Линейная регрессия", y_test, lr_pred)
results['Линейная регрессия'] = {'RMSE': rmse, 'MAE': mae,
                                  'y_pred': y_pred_lr}
print()

# ── 5. МОДЕЛЬ 2: СЛУЧАЙНЫЙ ЛЕС
print("=" * 55)
print("ШАГ 4: Случайный лес")
print("=" * 55)

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

rmse, mae, y_true, y_pred_rf = evaluate("Случайный лес", y_test, rf_pred)
results['Случайный лес'] = {'RMSE': rmse, 'MAE': mae,
                             'y_pred': y_pred_rf}
print()

# ── 6. МОДЕЛЬ 3: НЕЙРОСЕТЬ (MLP)
print("=" * 55)
print("ШАГ 5: Нейронная сеть (MLP)")
print("=" * 55)

def build_mlp(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu',
                           input_shape=(input_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

mlp_model = build_mlp(input_dim=X_train.shape[1])

history = mlp_model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True
        )
    ],
    verbose=1
)

mlp_pred = mlp_model.predict(X_test).flatten()
rmse, mae, y_true, y_pred_mlp = evaluate("MLP", y_test, mlp_pred)
results['MLP (нейросеть)'] = {'RMSE': rmse, 'MAE': mae,
                               'y_pred': y_pred_mlp}
print()

# ── 7. СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ
print("=" * 55)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 55)
print(f"{'Модель':<25} {'RMSE':>12} {'MAE':>12}")
print("-" * 50)
for name, vals in results.items():
    print(f"{name:<25} ${vals['RMSE']:>10,.2f} ${vals['MAE']:>10,.2f}")
print()

# ── 8. ГРАФИКИ
y_true_dollars = np.expm1(y_test.values)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Факт vs Прогноз (тестовая выборка)', fontsize=14)

model_names = ['Линейная регрессия', 'Случайный лес', 'MLP (нейросеть)']
preds       = [y_pred_lr, y_pred_rf, y_pred_mlp]
colors      = ['#3498db', '#2ecc71', '#e74c3c']

for ax, name, pred, color in zip(axes, model_names, preds, colors):
    ax.scatter(y_true_dollars, pred, alpha=0.3, s=10, color=color)
    lim = max(y_true_dollars.max(), pred.max())
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1, label='Идеальный прогноз')
    ax.set_xlabel('Факт ($)')
    ax.set_ylabel('Прогноз ($)')
    ax.set_title(name)
    ax.legend(fontsize=8)
    rmse_val = results[name]['RMSE']
    mae_val  = results[name]['MAE']
    ax.text(0.05, 0.92,
            f'RMSE=${rmse_val:,.0f}\nMAE=${mae_val:,.0f}',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('figure_3_1_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: figure_3_1_scatter.png")

# ── 9. СТОЛБЧАТАЯ ДИАГРАММА СРАВНЕНИЯ
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Сравнение моделей по метрикам качества', fontsize=13)

names = list(results.keys())
rmse_vals = [results[n]['RMSE'] for n in names]
mae_vals  = [results[n]['MAE']  for n in names]
bar_colors = ['#3498db', '#2ecc71', '#e74c3c']

ax1.bar(names, rmse_vals, color=bar_colors, edgecolor='white', linewidth=0.5)
ax1.set_title('RMSE (чем меньше — тем лучше)')
ax1.set_ylabel('Ошибка ($)')
for i, v in enumerate(rmse_vals):
    ax1.text(i, v + 5, f'${v:,.0f}', ha='center', fontsize=9)

ax2.bar(names, mae_vals, color=bar_colors, edgecolor='white', linewidth=0.5)
ax2.set_title('MAE (чем меньше — тем лучше)')
ax2.set_ylabel('Ошибка ($)')
for i, v in enumerate(mae_vals):
    ax2.text(i, v + 2, f'${v:,.0f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('figure_3_2_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: figure_3_2_comparison.png")

# ── 10. ВАЖНОСТЬ ПРИЗНАКОВ (СЛУЧАЙНЫЙ ЛЕС)
importances = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False).head(15)

fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind='barh', ax=ax, color='#2ecc71', edgecolor='white')
ax.set_title('Топ-15 важных признаков (Случайный лес)')
ax.set_xlabel('Важность признака')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('figure_3_3_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: figure_3_3_importance.png")

# ── 11. КРИВАЯ ОБУЧЕНИЯ (MLP)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history['loss'],     label='Ошибка на обучении',    color='#e74c3c')
ax.plot(history.history['val_loss'], label='Ошибка на валидации',   color='#3498db')
ax.set_title('Кривая обучения MLP')
ax.set_xlabel('Эпоха')
ax.set_ylabel('MSE (loss)')
ax.legend()
plt.tight_layout()
plt.savefig('figure_3_4_learning_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("График сохранён: figure_3_4_learning_curve.png")

print("\nВсе графики сохранены. Скрипт завершён.")
