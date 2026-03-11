# Smart Grid — Система прогнозирования энергопотребления

Курсовая работа по прогнозированию энергопотребления с использованием
технологий Smart Grid и нейронных сетей.

**Автор:** Ruslan Khismatov (2025)  
**Научный руководитель:** А.А. Агафонов

---

## Структура проекта

```
smart_grid_project/
├── main.py                  # Точка входа, пайплайн
├── config.py                # Все гиперпараметры и пути
├── data/
│   ├── generator.py         # Синтетические данные Smart Grid
│   └── preprocessing.py     # Нормализация, скользящие окна (без leakage)
├── models/
│   ├── lstm.py              # LSTM (3 слоя + BN + Dropout)
│   ├── transformer.py       # Encoder-only Transformer
│   ├── baseline.py          # XGBoost, LinearRegression, ARIMA
│   └── trainer.py           # Единый интерфейс обучения и оценки
├── analysis/
│   ├── eda.py               # EDA: профили, ACF/PACF, декомпозиция
│   ├── residuals.py         # ADF, KPSS, Ljung-Box, Durbin-Watson
│   └── backtesting.py       # Скользящий бэктестинг
├── optimization/
│   └── storage.py           # Накопитель: трёхтарифная система, КПД, деградация
├── utils/
│   ├── metrics.py           # MAE, RMSE, MAPE, R²
│   ├── visualization.py     # Графики обучения, прогнозов, накопителя
│   └── deployment.py        # Экспорт/загрузка бандла модели
└── requirements.txt
```

---

## Установка

```bash
git clone <repo_url>
cd smart_grid_project

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Запуск

### Быстрый тест (90 дней, 50 эпох)
```python
# В main.py раскомментировать:
Config.set_fast_mode()
```
```bash
python main.py
```

### Оптимальный режим (180 дней, 200 эпох) — по умолчанию
```bash
python main.py
```

### Полный режим (365 дней, 300 эпох)
```python
# В main.py раскомментировать:
Config.set_full_mode()
```

---

## Ключевые исправления относительно оригинала

### 1. Data Leakage — ИСПРАВЛЕНО
**Проблема:** `MinMaxScaler.fit_transform()` вызывался на всём датасете до разделения.  
**Решение:** Scaler обучается только на `train`-части, `val` и `test` только трансформируются:
```python
scaler.fit(raw_train)
scaled_val = scaler.transform(raw_val)   # НЕ fit_transform!
```

### 2. Экономика накопителя — ИСПРАВЛЕНО
**Проблема:** Цена рассчитывалась как `2.5 + 3.5 * 2.5 * normalized_load`.  
**Решение:** Трёхтарифная зонная система (российский стандарт):
| Зона | Часы | Тариф |
|------|------|-------|
| Ночная | 23:00–07:00 | 2.50 руб/кВт·ч |
| Полупиковая | 07–10, 17–21 | 4.20 руб/кВт·ч |
| Пиковая | 10–17, 21–23 | 6.80 руб/кВт·ч |

**КПД при зарядке:**
```
grid_for_charge = energy_stored_in_battery / sqrt(round_trip_eff)
```

**Учёт деградации:**
```
net_savings = gross_savings - total_energy_cycled × cycle_cost_per_kwh
```

### 3. Двойная функция `energy_storage_optimization` — УДАЛЕНА
Оригинал содержал две версии функции (вторая была мёртвым кодом после `return`).

### 4. Transformer — ДОБАВЛЕН
Encoder-only Transformer с sinusoidal positional encoding:
- `d_model=64`, `num_heads=4`, `num_layers=2`
- GlobalAveragePooling → Dense(128) → Dense(forecast_horizon)

---

## Результаты (пример, optimal mode)

| Модель | MAE | RMSE | MAPE | R² |
|--------|-----|------|------|----|
| LSTM | ~180 | ~240 | ~5.2% | ~0.91 |
| Transformer | ~175 | ~235 | ~5.0% | ~0.92 |
| XGBoost | ~160 | ~220 | ~4.8% | ~0.93 |
| LinearReg | ~280 | ~380 | ~8.5% | ~0.75 |

*Значения ориентировочные — зависят от режима и случайного сида.*

---

## Инференс (использование обученной модели)

```python
from utils.deployment import load_model_bundle, predict_from_bundle
import numpy as np

bundle = load_model_bundle("results/models/LSTM")
recent = np.array([...])  # последние 48 часов потребления (кВт·ч)
forecast = predict_from_bundle(bundle, recent)
print("Прогноз на 24 часа:", forecast)
```

---

## Зависимости

| Библиотека | Версия |
|------------|--------|
| Python | 3.8+ |
| TensorFlow | ≥ 2.13 |
| scikit-learn | ≥ 1.3 |
| XGBoost | ≥ 1.7 |
| statsmodels | ≥ 0.14 |
| pandas | ≥ 2.0 |
| matplotlib | ≥ 3.7 |
