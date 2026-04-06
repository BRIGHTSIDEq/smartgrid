# Smart Grid Forecasting — мульти-модельный пайплайн прогноза энергопотребления

Проект моделирует энергопотребление «умного города», обучает несколько классов моделей (LSTM, Transformer, PatchTST, Ridge, XGBoost), сравнивает их на едином бэктесте и использует лучший прогноз для оптимизации накопителя энергии.

---

## 1) Что делает проект (суть работы)

Система проходит полный цикл:

1. **Генерирует синтетические почасовые данные города** (погода, бытовая нагрузка, EV-зарядка, солнечная генерация, DSR, промышленность, аномалии).
2. **Готовит данные без leakage** (scaler обучается только на train, строятся лаги и оконные выборки для multi-horizon прогноза).
3. **Обучает набор моделей**:
   - TCN-BiLSTM-Attention (кастомная LSTM-модель),
   - Vanilla Transformer,
   - PatchTST,
   - Ridge (как LinearRegression-бейзлайн),
   - XGBoost с лаг- и rolling-признаками.
4. **Считает метрики и диагностику ошибок** (MAE/RMSE/MAPE/R² + ACF/DW/kurtosis).
5. **Выбирает лучшую модель**, строит визуализации и бэктестинг.
6. **Использует прогноз в задаче оптимизации накопителя** (тарифы, КПД, деградация, O&M, окупаемость).

---

## 2) Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Режимы запуска
В `main.py` переключите один из режимов:

- `Config.set_fast_mode()` — быстрый smoke run.
- `Config.set_optimal_mode()` — сбалансированный режим.
- `Config.set_full_mode()` — максимальное качество (долго, ресурсоёмко).

---

## 3) Структура проекта и роль каждого файла

> Ниже перечислены **рабочие исходники** (без `__pycache__`, артефактов в `results/`).

### Корень
- `main.py` — точка входа: orchestration всего пайплайна (генерация → обучение → сравнение → бэктест → storage).
- `config.py` — все гиперпараметры, режимы (`fast/optimal/full`), пути и логирование.
- `requirements.txt` — зависимости проекта.
- `README.md` — текущая документация.
- `conftest.py` — конфигурация тестового окружения (pytest).
- `test_model_stability.py` — тест/проверка стабильности модели.
- `thesis_section_3_transformers.md` — исследовательская текстовая секция по Transformer.

### `data/`
- `data/__init__.py` — package marker.
- `data/generator.py` — генерация синтетического smart-grid датасета + валидация правдоподобия.
- `data/preprocessing.py` — scaler, инженерия признаков, лаги, оконные выборки `(X, Y)`, inverse scaling, integrity checks.

### `models/`
- `models/__init__.py` — package marker.
- `models/lstm.py` — TCN-BiLSTM-Attention с seasonal skip и Keras3-safe RevIN-нормализацией.
- `models/transformer.py` — VanillaTransformer, PatchTST (+ общие компоненты внимания/позиционных кодировок).
- `models/baseline.py` — Ridge baseline и XGBoost baseline с лаг/rolling-фичами.
- `models/trainer.py` — общий trainer, callbacks, OOM-safe batch fallback, evaluation, residual diagnostics.

### `analysis/`
- `analysis/__init__.py` — package marker.
- `analysis/eda.py` — EDA: профили, сезонность, ACF/PACF, декомпозиции.
- `analysis/residuals.py` — статистика остатков (ADF/KPSS/Ljung-Box/JB/DW).
- `analysis/backtesting.py` — скользящий backtesting по окнам.

### `optimization/`
- `optimization/__init__.py` — package marker.
- `optimization/storage.py` — оптимизация работы накопителя и сравнение стратегий.

### `utils/`
- `utils/__init__.py` — package marker.
- `utils/metrics.py` — MAE, RMSE, MAPE, R² и вспомогательные метрики.
- `utils/visualization.py` — графики обучения/сравнений/прогнозов/storage.
- `utils/attention_visualization.py` — визуализация attention-карт и head specialization.
- `utils/deployment.py` — экспорт/загрузка model bundle для инференса.

### `experiments/`
- `experiments/__init__.py` — package marker.
- `experiments/transformer_ablation.py` — абляции Transformer-конфигураций.

---

## 4) Как генерируются данные (подробно)

Генератор строит почасовой ряд длиной `days * 24` и формирует многокомпонентное потребление:

## 4.1 Календарь и сезонность
- Час, день недели, день года, выходные/праздники/новогодний период.
- Зимний буст и летний дип, плюс медленный тренд.

## 4.2 Погода
- Температура: годовая + суточная гармоники + AR-шум.
- Волны жары/холода добавляются событиями блоками по нескольку суток.
- Влажность, облачность, ветер — отдельные процессы с собственной динамикой.

## 4.3 Бытовое потребление
- Три типа домохозяйств: early birds / standard / night owls.
- Для каждого типа профиль собирается из суммы Gaussian peaks (утро/день/вечер).
- Праздники и выходные смещают форму профиля.

## 4.4 Городская структура (district simulation)
- Город разбивается на `city_districts` районов.
- Районы получают веса (Dirichlet), офисный/жилой bias и амплитуды commute peak.
- Суммарная `district_curve` умножает базовое потребление.

**Пример:**
- если район «деловой», утром (8–10) вклад выше;
- если район «спальный», вечерний пик сильнее;
- в выходные деловой район «проседает», спальный — меньше.

## 4.5 EV, Solar, DSR, Industrial
- **EV**: домашняя/публичная зарядка, холодовые surges, пятничные кластеры, коммерческий флот.
- **Solar**: дневной колокол, сезонность и облачность.
- **DSR**: редкие события снижения нагрузки в пиковые часы.
- **Industrial**: заводы с режимами смен и maintenance-провалами.

## 4.6 Финальная формула (концептуально)

`consumption ≈ base_profile × weather_response × district_curve × anomalies + EV + Industrial - Solar - DSR + cascade`

Итогом является датафрейм с `timestamp`, `consumption` и ковариатами, пригодный для supervised learning.

---

## 5) Подготовка признаков и target

`data/preprocessing.py` формирует:

1. **Лаги потребления**: `24h`, `48h`, `168h`.
2. **Календарные признаки**: sin/cos часа, sin/cos дня недели, сезонные синусы.
3. **Погодные признаки**: нормализованные `temperature/humidity/wind/cloud` + `temperature_squared`.
4. **Сигналы системы**: EV norm, solar norm, DSR active, rolling mean/std.
5. **Оконные выборки**:
   - вход `X`: `(history_length, n_features)`,
   - цель `Y`: вектор из `forecast_horizon` будущих часов.

Важно: scaler fit выполняется **только на train**.

---

## 6) Модели и почему они устроены именно так

## 6.1 TCN-BiLSTM-Attention (`models/lstm.py`)

**Архитектура:**
- входное окно `(T, F)`;
- RevIN-нормализация канала consumption (Keras3-safe через `Lambda`);
- multi-dilation TCN-блоки (локальные паттерны и лаги разной длины);
- BiLSTM (последовательная динамика);
- Temporal Multi-Head Attention (фокус на релевантных шагах);
- Dense head;
- SeasonalSkip: смешивание нейросетевого прогноза с seasonal naive (вчера в это же время).

**Почему так:**
- TCN ловит локальные всплески;
- LSTM — гладкую эволюцию последовательности;
- attention позволяет «вспоминать» важные участки окна;
- seasonal skip стабилизирует multi-horizon прогноз и ускоряет сходимость.

## 6.2 Vanilla Transformer (`models/transformer.py`)

**Архитектура:**
- input projection → positional encoding;
- stack encoder blocks (Pre-LN, MHA, FFN, stochastic depth);
- pooled context + last token;
- MLP head;
- optional seasonal residual blend.

**Почему так:**
- self-attention удобен для дальних зависимостей (например, суточные/недельные структуры);
- stochastic depth снижает переобучение глубокого энкодера;
- seasonal residual помогает на strongly-seasonal данных.

## 6.3 PatchTST (`models/transformer.py`)

**Идея:** вместо отдельных часов модель видит «патчи» окна (как слова в NLP).

**Почему это работает:**
- сокращает длину последовательности для attention;
- усиливает устойчивость к шуму;
- хорошо переносится на long-horizon forecasting.

## 6.4 Baselines (`models/baseline.py`)

- **Ridge (LinearRegression wrapper):** разворачивает окно в плоский вектор.
- **XGBoost:** строит агрегированные lag/rolling фичи по каналам и обучает multi-horizon регрессию.

**Зачем baseline:**
- честный ориентир качества;
- проверка, дают ли нейросети реальный прирост.

---

## 7) Как система избегает частых проблем

- **Data leakage**: строго train-only fit для scaler.
- **Keras3 graph errors**: RevIN через Keras-слои/Lambda.
- **OOM при обучении**: trainer автоматически уменьшает batch size при `ResourceExhausted`.
- **Нестабильная валидация**: callbacks (EarlyStopping + ReduceLROnPlateau).

---

## 8) Какие артефакты формируются

После запуска в `results/` сохраняются:
- `results/models/` — модели и scaler’ы,
- `results/plots/` — графики обучения, сравнения, внимания, остатков, бэктеста, storage,
- `results/logs/run.log` — полный журнал пайплайна.

---

## 9) Пример практического применения

### Сценарий: диспетчер энергосистемы на сутки вперёд
1. Берём последние `history_length` часов телеметрии.
2. Делаем прогноз на `forecast_horizon=24`.
3. Передаём прогноз в оптимизатор накопителя.
4. Получаем график charge/discharge и оценку экономии/окупаемости.

Это можно использовать для:
- снижения peak-load,
- арбитража по тарифным зонам,
- планирования DSR-событий,
- оценки бизнес-эффекта BESS.

---

## 10) Рекомендации по запуску

- Для разработки: `fast_mode`.
- Для основного отчёта: `optimal_mode`.
- Для максимального качества и ночных прогонов: `full_mode` (требует больше RAM/времени).

Если есть GPU с ограниченной памятью, оставьте adaptive OOM fallback в trainer включённым (по умолчанию).

---

## 11) Технологический стек

- Python 3.10+
- TensorFlow / Keras
- NumPy, Pandas
- scikit-learn
- XGBoost
- Matplotlib / Seaborn
- Statsmodels

