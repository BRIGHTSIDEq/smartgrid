# -*- coding: utf-8 -*-
"""
config.py — Единая точка настройки пайплайна.

Все гиперпараметры, пути и режимы прогона собраны здесь: ни один модуль не
содержит зашитых констант обучения. Режим выбирается флагом --mode у main.py.

Режимы прогона:
    smoke   ~1 минута   — проверка работоспособности пайплайна целиком.
                          Данные и модели минимальны, результаты бессмысленны.
    fast    ~15 минут   — черновые эксперименты, отладка гипотез.
    optimal ~2 часа     — рабочий режим, пригодный для итоговых таблиц.
    full    ~10 часов   — максимальное качество: 2 года данных, история 192 ч.

Параметры генератора и экономики откалиброваны по фактическим данным для
России (Москва). Эталонные значения и источники — в разделе RealWorldReference
ниже; соответствие проверяется функцией
data.generator.validate_against_reference() при каждом прогоне.

История изменений — в CHANGELOG.md.
"""

import os
import logging


class RealWorldReference:
    """
    Эталонные значения реального мира, по которым калибруется генератор.

    Все цифры относятся к России (Московский регион) и служат двум целям:
    задают параметры генератора и используются как контрольные значения при
    автоматической сверке сгенерированного ряда с действительностью.

    ИСТОЧНИКИ
      Тарифные зоны и ставки — АО «Мосэнергосбыт», трёхзонный тариф для
        населения Москвы, дома с газовыми плитами, с 01.07.2025.
        Зоны установлены единым для РФ порядком: пиковая 07–10 и 17–21,
        полупиковая 10–17 и 21–23, ночная 23–07.
      Потребление домохозяйства — оценка НИУ ВШЭ: 206 кВт·ч в месяц
        на домохозяйство в среднем по России.
      Климат Москвы — климатические нормы: январь −6.5 °C, июль +19.0 °C.
      Сезонность нагрузки — ЕЭС России: январское потребление примерно
        на четверть выше июльского.
      Доля электромобилей — рынок РФ 2025: около 1% продаж новых автомобилей,
        доля в парке существенно ниже.
      Стоимость накопителя — мировой рынок BESS 2025: 200–400 $/кВт·ч за
        систему «под ключ»; для РФ принята верхняя часть диапазона.
      Ресурс LFP — 6000+ циклов до 70–80% остаточной ёмкости.
    """

    # ── Потребление ──────────────────────────────────────────────────────────
    KWH_PER_HOUSEHOLD_MONTH: float = 206.0      # среднее по РФ
    KWH_PER_HOUSEHOLD_MONTH_RANGE = (150.0, 300.0)

    # Доля непромышленных и промышленных потребителей сверх населения.
    # Для распределительного фидера городского района коммерческая и мелкая
    # промышленная нагрузка сопоставима с бытовой.
    NONRESIDENTIAL_SHARE: float = 0.45
    NONRESIDENTIAL_SHARE_RANGE = (0.20, 0.90)

    # ── Климат Москвы ────────────────────────────────────────────────────────
    TEMP_JANUARY_MEAN: float = -6.5
    TEMP_JULY_MEAN: float = 19.0
    TEMP_ABS_MIN: float = -35.0                  # практический минимум
    TEMP_ABS_MAX: float = 38.0                   # рекорд 2010 года ≈ +38.2

    @classmethod
    def temp_annual_mean(cls) -> float:
        return (cls.TEMP_JULY_MEAN + cls.TEMP_JANUARY_MEAN) / 2.0

    @classmethod
    def temp_annual_amplitude(cls) -> float:
        return (cls.TEMP_JULY_MEAN - cls.TEMP_JANUARY_MEAN) / 2.0

    # ── Сезонность и суточный профиль ────────────────────────────────────────
    WINTER_SUMMER_RATIO: float = 1.25
    WINTER_SUMMER_RATIO_RANGE = (1.10, 1.45)
    WEEKEND_WEEKDAY_RATIO_RANGE = (0.88, 0.99)
    # Коэффициент заполнения графика (среднее / максимум) сильно зависит от
    # уровня агрегации: для энергосистемы в целом это 0.70–0.85, для смешанного
    # городского фидера 0.50–0.70, для чисто бытового — 0.40–0.55. Модель
    # описывает районный фидер, где на бытовую нагрузку приходится примерно
    # половина потребления, поэтому диапазон задан соответственно.
    LOAD_FACTOR_RANGE = (0.42, 0.75)
    CV_RANGE = (0.12, 0.35)                      # коэффициент вариации
    MORNING_PEAK_HOURS = (7, 11)
    EVENING_PEAK_HOURS = (17, 22)
    ACF24_MIN: float = 0.50

    # ── Тарифы, руб/кВт·ч (Москва, с 01.07.2025, газовые плиты) ─────────────
    TARIFF_PEAK: float = 11.24
    TARIFF_HALF_PEAK: float = 7.87
    TARIFF_NIGHT: float = 4.08

    # ── Накопитель ───────────────────────────────────────────────────────────
    BATTERY_CAPEX_RUB_PER_KWH: float = 25_000.0  # ≈300 $/кВт·ч под ключ
    BATTERY_CYCLE_LIFE: int = 6_000              # циклов до 80% ёмкости (LFP)
    BATTERY_REF_DOD: float = 0.80                # глубина разряда для ресурса
    BATTERY_ROUND_TRIP_EFF: float = 0.88         # с учётом инвертора

    # Типоразмер накопителя задаётся не абсолютной величиной, а долей от пика
    # нагрузки: батарея мощностью почти в размер сети физически бессмысленна и
    # делает экономику заведомо убыточной за счёт капитальных затрат.
    # Практика сетевых BESS: мощность 15–25% пика, длительность 2–4 часа.
    BATTERY_POWER_SHARE_OF_PEAK: float = 0.20
    BATTERY_DURATION_HOURS: float = 4.0
    TYPICAL_LOAD_FACTOR: float = 0.48            # среднее / пик, см. LOAD_FACTOR_RANGE

    # ── Проникновение технологий Smart Grid ──────────────────────────────────
    # «current» — состояние на 2025 год, «forward» — перспективный сценарий.
    # Разница между сценариями показывает, при каком уровне проникновения
    # распределённые ресурсы начинают заметно влиять на профиль нагрузки.
    SCENARIOS = {
        "current": {
            "ev_penetration": 0.005,     # ≈0.5% домохозяйств, оценка сверху
            "solar_penetration": 0.002,  # микрогенерация в РФ единична
            "dsr_events_per_year": 10,
            "dsr_strength": (0.02, 0.06),
        },
        "forward": {
            "ev_penetration": 0.15,
            "solar_penetration": 0.030,
            "dsr_events_per_year": 20,
            "dsr_strength": (0.05, 0.12),
        },
    }

    # ── Электротранспорт: мощность зарядки, кВт ──────────────────────────────
    EV_HOME_POWER_KW = (3.5, 7.4)                # 16 А и 32 А однофазные
    EV_PUBLIC_POWER_KW = (22.0, 50.0)
    EV_FLEET_POWER_KW = (30.0, 60.0)

    # ── Прочее ───────────────────────────────────────────────────────────────
    ANNUAL_TREND: float = 0.015                  # рост потребления ~1.5%/год
    SOLAR_PANEL_PEAK_KW: float = 5.0             # типовая бытовая установка


class Config:

    SEED: int = 42
    DAYS: int = 730
    HOUSEHOLDS: int = 2500
    START_DATE: str = "2024-01-01"
    N_FEATURES: int = 26

    HISTORY_LENGTH: int = 48
    FORECAST_HORIZON: int = 24
    STORAGE_HORIZON: int = 720

    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15

    EPOCHS: int = 240
    BATCH_SIZE: int = 32
    PATIENCE: int = 25
    LR_PATIENCE: int = 10
    LR_FACTOR: float = 0.5
    MIN_DELTA: float = 0.0

    # LSTM (TCN + BiLSTM + Attention)
    LSTM_UNITS_1: int = 96
    LSTM_UNITS_2: int = 64
    LSTM_UNITS_3: int = 64
    DROPOUT_RATE: float = 0.12
    LSTM_LEARNING_RATE: float = 2.0e-4
    LSTM_ATTN_HEADS: int = 4
    LSTM_USE_COSINE_DECAY: bool = False
    LSTM_TCN_FILTERS: int = 48
    LSTM_HUBER_DELTA: float = 0.05
    LSTM_SEASONAL_BLEND_INIT: float = 0.35

    # Transformer
    TRANSFORMER_D_MODEL: int = 192
    TRANSFORMER_N_HEADS: int = 8
    TRANSFORMER_N_LAYERS: int = 5
    TRANSFORMER_DFF: int = 384
    TRANSFORMER_DROPOUT: float = 0.10
    TRANSFORMER_LEARNING_RATE: float = 2e-4
    VANILLA_TRANSFORMER_LR: float = 7e-5
    TRANSFORMER_STOCHASTIC_DEPTH: float = 0.06
    PATCHTST_USE_REVIN: bool = True
    # Линейный прогрев с последующим косинусным затуханием — стандартное
    # расписание для трансформеров. Без прогрева обучение на первых шагах
    # делает крупные шаги и уходит в плохой минимум.
    TRANSFORMER_USE_WARMUP_COSINE: bool = True
    TRANSFORMER_WARMUP_FRACTION: float = 0.05
    VANILLA_USE_SEASONAL_RESIDUAL: bool = True
    VANILLA_SEASONAL_BLEND_INIT: float = 0.40
    VANILLA_HUBER_DELTA: float = 0.05

    # XGBoost
    XGB_N_ESTIMATORS: int = 500
    XGB_MAX_DEPTH: int = 5
    XGB_LR: float = 0.05
    XGB_SUBSAMPLE: float = 0.80
    XGB_COLSAMPLE: float = 0.40

    # ── Генератор данных (калибровка — см. RealWorldReference) ──────────────
    GEN_SCENARIO: str = "current"

    # Масштаб нагрузки задаётся не абстрактным множителем, а фактическим
    # среднемесячным потреблением домохозяйства: так ряд остаётся
    # сопоставимым с реальностью при любом числе домохозяйств.
    GEN_KWH_PER_HOUSEHOLD_MONTH: float = RealWorldReference.KWH_PER_HOUSEHOLD_MONTH
    GEN_NONRESIDENTIAL_SHARE: float = RealWorldReference.NONRESIDENTIAL_SHARE

    GEN_TEMP_ANNUAL_MEAN: float = RealWorldReference.temp_annual_mean()
    GEN_TEMP_ANNUAL_AMPLITUDE: float = RealWorldReference.temp_annual_amplitude()
    GEN_TEMP_MIN: float = RealWorldReference.TEMP_ABS_MIN
    GEN_TEMP_MAX: float = RealWorldReference.TEMP_ABS_MAX

    # Отклик на температуру асимметричен: в России отопление преимущественно
    # центральное или газовое, а кондиционирование распространено слабо,
    # поэтому холодная ветвь заметно сильнее тёплой.
    GEN_TEMP_SETPOINT: float = 18.0              # порог включения отопления
    GEN_COOLING_SETPOINT: float = 24.0           # порог включения охлаждения
    GEN_HEATING_COEF: float = 2.0e-4
    GEN_COOLING_COEF: float = 1.2e-4

    GEN_HUMIDITY_THRESHOLD: float = 60.0
    GEN_HUMIDITY_COEF: float = 0.10
    GEN_WIND_TEMP_THRESHOLD: float = 10.0
    GEN_WIND_COEF: float = 0.05
    GEN_EARLY_BIRD_FRAC: float = 0.28
    GEN_NIGHT_OWL_FRAC:  float = 0.20
    GEN_AR_PHI: float   = 0.65
    GEN_AR_SIGMA: float = 0.030
    # Остаточная сезонность сверх температурного отклика — небольшая, иначе
    # отношение зима/лето уходит далеко за фактические 1.25.
    GEN_SEASONAL_WINTER_BOOST: float = 0.06
    GEN_SEASONAL_SUMMER_DIP:   float = 0.04
    GEN_ANNUAL_TREND: float = RealWorldReference.ANNUAL_TREND
    GEN_INDUSTRIAL_LOADS: int   = 8
    GEN_CITY_DISTRICTS: int = 12

    # Заполняются из сценария в apply_scenario()
    GEN_EV_PENETRATION: float = 0.005
    GEN_SOLAR_PENETRATION: float = 0.002
    GEN_DSR_EVENTS_PER_YEAR: int = 10
    GEN_DSR_STRENGTH = (0.02, 0.06)

    # ── Накопитель энергии ──────────────────────────────────────────────────
    BATTERY_CAPACITY: float = 4_500.0
    BATTERY_MAX_POWER: float = 2_250.0            # 0.5C — типично для сетевых BESS
    BATTERY_EFFICIENCY: float = RealWorldReference.BATTERY_ROUND_TRIP_EFF
    BATTERY_OM_SHARE: float = 0.015
    DEMAND_CHARGE_RUB_PER_KW_MONTH: float = 950.0
    BATTERY_MIN_SOC: float = 0.25
    BATTERY_MAX_SOC: float = 0.75

    # Производные величины (пересчитываются в _derive_battery_economics)
    BATTERY_COST_RUB: float = 0.0
    BATTERY_CYCLE_COST: float = 0.0

    TARIFF_PEAK: float = RealWorldReference.TARIFF_PEAK
    TARIFF_HALF_PEAK: float = RealWorldReference.TARIFF_HALF_PEAK
    TARIFF_NIGHT: float = RealWorldReference.TARIFF_NIGHT

    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR: str  = os.path.join(BASE_DIR, "results")
    MODELS_DIR: str  = os.path.join(BASE_DIR, "results", "models")
    PLOTS_DIR: str   = os.path.join(BASE_DIR, "results", "plots")
    LOGS_DIR: str    = os.path.join(BASE_DIR, "results", "logs")

    LOG_LEVEL: int = logging.INFO
    LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    LOG_DATE_FMT: str = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def create_dirs(cls):
        for d in (cls.OUTPUT_DIR, cls.MODELS_DIR, cls.PLOTS_DIR, cls.LOGS_DIR):
            os.makedirs(d, exist_ok=True)

    # ── Производные параметры ────────────────────────────────────────────────

    @classmethod
    def apply_scenario(cls, name: str = None):
        """
        Применяет сценарий проникновения технологий Smart Grid.

        "current" — фактическое состояние на 2025 год: электромобилей и
        микрогенерации в России пока единицы процентов, поэтому их вклад в
        профиль нагрузки мал. "forward" — перспективный сценарий, в котором
        распределённые ресурсы заметно меняют форму суточного графика.

        Сценарий влияет только на генератор данных и всегда фиксируется
        в метаданных прогона, чтобы результаты нельзя было перепутать.
        """
        name = name or cls.GEN_SCENARIO
        if name not in RealWorldReference.SCENARIOS:
            raise ValueError(
                f"Неизвестный сценарий {name!r}. "
                f"Доступны: {', '.join(RealWorldReference.SCENARIOS)}"
            )
        params = RealWorldReference.SCENARIOS[name]
        cls.GEN_SCENARIO = name
        cls.GEN_EV_PENETRATION = params["ev_penetration"]
        cls.GEN_SOLAR_PENETRATION = params["solar_penetration"]
        cls.GEN_DSR_EVENTS_PER_YEAR = params["dsr_events_per_year"]
        cls.GEN_DSR_STRENGTH = params["dsr_strength"]
        return cls

    @classmethod
    def expected_peak_load_kw(cls) -> float:
        """
        Оценка пиковой нагрузки сети по числу домохозяйств.

        Служит для типоразмера накопителя: сначала считается средняя нагрузка
        (бытовая плюс непромышленная), затем пик через коэффициент заполнения.
        """
        hours_per_month = 8766.0 / 12.0
        residential = cls.HOUSEHOLDS * cls.GEN_KWH_PER_HOUSEHOLD_MONTH / hours_per_month
        mean_total = residential * (1.0 + cls.GEN_NONRESIDENTIAL_SHARE)
        return mean_total / RealWorldReference.TYPICAL_LOAD_FACTOR

    @classmethod
    def _derive_battery_economics(cls):
        """
        Подбирает типоразмер накопителя под нагрузку и пересчитывает экономику.

        РАЗМЕР. Мощность и ёмкость масштабируются от пика сети, а не задаются
        константой. Иначе при изменении числа домохозяйств батарея оказывается
        либо ничтожной, либо сопоставимой по мощности со всей сетью — во втором
        случае капитальные затраты гарантированно перекрывают любую экономию,
        и расчёт окупаемости теряет смысл.

        СТОИМОСТЬ ДЕГРАДАЦИИ — не произвольная константа, а следствие
        капитальных затрат и ресурса: каждый кВт·ч, прошедший через батарею,
        расходует часть её жизненного цикла.

            cycle_cost = CAPEX / (ёмкость × глубина разряда × ресурс в циклах)

        При 25 000 руб/кВт·ч, DoD 80% и 6000 циклах получается около
        5.2 руб/кВт·ч — величина, сопоставимая с тарифным спредом. Занижение
        этого параметра делает арбитраж искусственно выгодным.
        """
        ref = RealWorldReference
        peak = cls.expected_peak_load_kw()
        cls.BATTERY_MAX_POWER = round(peak * ref.BATTERY_POWER_SHARE_OF_PEAK, 1)
        cls.BATTERY_CAPACITY = round(cls.BATTERY_MAX_POWER * ref.BATTERY_DURATION_HOURS, 1)

        cls.BATTERY_COST_RUB = ref.BATTERY_CAPEX_RUB_PER_KWH * cls.BATTERY_CAPACITY
        throughput = cls.BATTERY_CAPACITY * ref.BATTERY_REF_DOD * ref.BATTERY_CYCLE_LIFE
        cls.BATTERY_CYCLE_COST = cls.BATTERY_COST_RUB / throughput
        return cls

    @classmethod
    def finalize(cls, scenario: str = None):
        """Применяет сценарий и пересчитывает производные параметры."""
        cls.apply_scenario(scenario)
        cls._derive_battery_economics()
        return cls

    @staticmethod
    def _make_console_encoding_safe():
        """
        Гарантирует, что вывод в консоль не уронит программу на кодировке.

        Логи содержат кириллицу и псевдографику (─ ═ →), которых нет в
        однобайтовых кодовых страницах. На русской Windows консоль по умолчанию
        работает в cp1251, и первая же такая строка вызывает
        UnicodeEncodeError: 'charmap' codec can't encode character.

        Поток переводится в UTF-8; если терминал этого не поддерживает,
        включается замена непредставимых символов, чтобы вывод деградировал
        до «?» вместо аварийного завершения. Пользователю не нужно
        самостоятельно выставлять PYTHONIOENCODING.
        """
        import sys
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is None:
                continue
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                try:
                    reconfigure(errors="replace")
                except Exception:
                    pass

    @classmethod
    def setup_logging(cls):
        cls.create_dirs()
        cls._make_console_encoding_safe()
        logging.basicConfig(
            level=cls.LOG_LEVEL, format=cls.LOG_FORMAT, datefmt=cls.LOG_DATE_FMT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(cls.LOGS_DIR, "run.log"), encoding="utf-8"),
            ],
        )
        return logging.getLogger("smart_grid")

    # ── Многорядные (panel) режимы ──────────────────────────────────────────
    # Отдельная группа параметров: panel меняет не объём одного ряда, а число
    # рядов, поэтому переиспользовать DAYS/HOUSEHOLDS было бы неверно.
    PANEL_CITIES: int = 1
    PANEL_FEEDERS_PER_CITY: int = 4
    PANEL_DAYS: int = 90
    PANEL_EPOCHS: int = 30
    PANEL_BATCH_SIZE: int = 128
    PANEL_PATIENCE: int = 8
    PANEL_HISTORY: int = 48
    PANEL_XGB_ESTIMATORS: int = 120
    PANEL_DLINEAR_UNITS: int = 64
    PANEL_DLINEAR_LR: float = 2e-3
    PANEL_WINDOW_STRIDE: int = 1

    @classmethod
    def set_panel_smoke_mode(cls):
        """Минимальная панель для проверки работоспособности конвейера."""
        cls.PANEL_CITIES = 1; cls.PANEL_FEEDERS_PER_CITY = 4; cls.PANEL_DAYS = 90
        cls.PANEL_EPOCHS = 20; cls.PANEL_PATIENCE = 6
        cls.PANEL_HISTORY = 48; cls.PANEL_XGB_ESTIMATORS = 60
        cls.PANEL_WINDOW_STRIDE = 1
        cls.FORECAST_HORIZON = 24
        logging.getLogger("smart_grid").info(
            "Panel smoke: %d город × %d фидера × %d дней",
            cls.PANEL_CITIES, cls.PANEL_FEEDERS_PER_CITY, cls.PANEL_DAYS)

    @classmethod
    def set_panel_fast_mode(cls):
        """Рабочая панель: несколько городов, год истории."""
        cls.PANEL_CITIES = 2; cls.PANEL_FEEDERS_PER_CITY = 8; cls.PANEL_DAYS = 365
        cls.PANEL_EPOCHS = 60; cls.PANEL_PATIENCE = 10
        cls.PANEL_HISTORY = 48; cls.PANEL_XGB_ESTIMATORS = 200
        cls.PANEL_WINDOW_STRIDE = 1
        cls.FORECAST_HORIZON = 24
        logging.getLogger("smart_grid").info(
            "Panel fast: %d города × %d фидеров × %d дней",
            cls.PANEL_CITIES, cls.PANEL_FEEDERS_PER_CITY, cls.PANEL_DAYS)

    @classmethod
    def set_panel_optimal_mode(cls):
        """
        Крупная панель. Требует потоковой подачи данных.

        Смысл режима — не длина окна, а состав панели: 96 рядов вместо 16 и три
        года вместо одного. Второе снимает ограничение panel-fast, где при
        единственном годе на тест приходился сезон, отсутствовавший в обучении.

        Все окна со сдвигом 1 час дали бы 2.5 млн штук и порядка 40 ГБ. Окна
        прореживаются шагом 11: соседние перекрываются на 47 часов из 48 и
        почти дублируют друг друга, поэтому потеря сведений мала, а расход
        памяти падает на порядок. Шаг взаимно прост с 24 и 168, поэтому начала
        окон равномерно обходят все часы суток и дни недели — при шаге 4
        обучение видело бы прогнозы, начинающиеся только в часы 0, 4, 8, 12,
        16 и 20.

        Длина истории оставлена как в panel-fast: тогда между режимами
        различаются только состав панели и длительность, и разницу в
        результатах можно отнести к ним, а не к изменённому окну. Недельная
        зависимость и так доступна моделям через канал lag_168h.

        Это осознанное упрощение, а не полноценная потоковая подача: она нужна,
        чтобы обучаться на всех окнах без прореживания.
        """
        cls.PANEL_CITIES = 4; cls.PANEL_FEEDERS_PER_CITY = 24
        cls.PANEL_DAYS = 365 * 3
        cls.PANEL_EPOCHS = 120; cls.PANEL_PATIENCE = 15
        cls.PANEL_HISTORY = 48; cls.PANEL_XGB_ESTIMATORS = 400
        cls.PANEL_WINDOW_STRIDE = 11
        cls.FORECAST_HORIZON = 24
        logging.getLogger("smart_grid").info(
            "Panel optimal: %d городов × %d фидеров × %d дней, шаг окон %d",
            cls.PANEL_CITIES, cls.PANEL_FEEDERS_PER_CITY, cls.PANEL_DAYS,
            cls.PANEL_WINDOW_STRIDE)

    @classmethod
    def set_smoke_mode(cls):
        """
        Минимальный прогон для проверки работоспособности пайплайна.

        Назначение — убедиться, что все стадии отрабатывают без ошибок и
        совместимы по формам данных. Метрики в этом режиме интерпретации
        не подлежат: 30 дней данных и 2 эпохи обучения.
        """
        cls.DAYS = 40; cls.HOUSEHOLDS = 60; cls.EPOCHS = 2
        cls.PATIENCE = 2; cls.LR_PATIENCE = 1
        cls.HISTORY_LENGTH = 48; cls.FORECAST_HORIZON = 24
        cls.STORAGE_HORIZON = 240; cls.N_FEATURES = 26
        cls.BATCH_SIZE = 64
        cls.LSTM_UNITS_1 = 16; cls.LSTM_UNITS_2 = 16; cls.LSTM_UNITS_3 = 16
        cls.LSTM_ATTN_HEADS = 2; cls.LSTM_TCN_FILTERS = 8
        cls.DROPOUT_RATE = 0.10; cls.LSTM_LEARNING_RATE = 1e-3
        cls.LSTM_USE_COSINE_DECAY = False
        cls.TRANSFORMER_D_MODEL = 16; cls.TRANSFORMER_N_HEADS = 2
        cls.TRANSFORMER_N_LAYERS = 1; cls.TRANSFORMER_DFF = 32
        cls.TRANSFORMER_DROPOUT = 0.10; cls.TRANSFORMER_LEARNING_RATE = 1e-3
        cls.VANILLA_TRANSFORMER_LR = 1e-3; cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.0
        cls.XGB_N_ESTIMATORS = 20
        cls.GEN_INDUSTRIAL_LOADS = 2; cls.GEN_CITY_DISTRICTS = 3
        logging.getLogger("smart_grid").warning(
            "SMOKE MODE: %d дней, %d эпох — только проверка работоспособности, "
            "метрики интерпретации не подлежат.", cls.DAYS, cls.EPOCHS,
        )

    @classmethod
    def set_fast_mode(cls):
        cls.DAYS = 365; cls.HOUSEHOLDS = 250; cls.EPOCHS = 120
        cls.PATIENCE = 20; cls.LR_PATIENCE = 8
        cls.HISTORY_LENGTH = 48; cls.STORAGE_HORIZON = 720; cls.N_FEATURES = 26
        cls.LSTM_UNITS_1 = 48; cls.LSTM_UNITS_2 = 48; cls.LSTM_UNITS_3 = 48
        cls.LSTM_ATTN_HEADS = 4; cls.LSTM_TCN_FILTERS = 32
        cls.DROPOUT_RATE = 0.25; cls.LSTM_LEARNING_RATE = 2e-4; cls.LSTM_USE_COSINE_DECAY = False
        cls.LSTM_SEASONAL_BLEND_INIT = 0.30; cls.LSTM_HUBER_DELTA = 0.05
        cls.TRANSFORMER_D_MODEL = 64; cls.TRANSFORMER_N_HEADS = 4
        cls.TRANSFORMER_N_LAYERS = 2; cls.TRANSFORMER_DFF = 128
        cls.TRANSFORMER_DROPOUT = 0.20; cls.TRANSFORMER_LEARNING_RATE = 3e-4
        cls.VANILLA_TRANSFORMER_LR = 8e-5; cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.05
        cls.PATCHTST_USE_REVIN = True
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True; cls.VANILLA_SEASONAL_BLEND_INIT = 0.35
        cls.VANILLA_HUBER_DELTA = 0.05; cls.XGB_N_ESTIMATORS = 300
        cls.GEN_INDUSTRIAL_LOADS = 4; cls.GEN_CITY_DISTRICTS = 8
        logging.getLogger("smart_grid").info(
            "Fast mode: DAYS=%d HH=%d EPOCHS=%d HIST=%d | LSTM BiLSTM=%d TCN=%d | VanTr LR=%.0e",
            cls.DAYS, cls.HOUSEHOLDS, cls.EPOCHS, cls.HISTORY_LENGTH,
            cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS, cls.VANILLA_TRANSFORMER_LR)

    @classmethod
    def set_optimal_mode(cls):
        # Усиленный режим для достижения более высокого R² у seq-моделей.
        cls.DAYS = 730; cls.HOUSEHOLDS = 2500; cls.EPOCHS = 240
        cls.PATIENCE = 25; cls.LR_PATIENCE = 10
        cls.HISTORY_LENGTH = 48; cls.STORAGE_HORIZON = 720; cls.N_FEATURES = 26
        # Ёмкость сокращена по результатам прогона: при 1.68 млн параметров
        # PatchTST достигал минимума валидации на 5-й эпохе из 240 и далее
        # только переобучался. Ёмкость — такой же гиперпараметр, как alpha у
        # Ridge или число деревьев у бустинга, и она должна быть соразмерна
        # объёму выборки.
        cls.LSTM_UNITS_1 = 48; cls.LSTM_UNITS_2 = 48; cls.LSTM_UNITS_3 = 48
        cls.LSTM_ATTN_HEADS = 4; cls.LSTM_TCN_FILTERS = 24
        cls.DROPOUT_RATE = 0.12; cls.LSTM_LEARNING_RATE = 2.0e-4; cls.LSTM_USE_COSINE_DECAY = False
        cls.LSTM_SEASONAL_BLEND_INIT = 0.35; cls.LSTM_HUBER_DELTA = 0.05
        cls.TRANSFORMER_D_MODEL = 64; cls.TRANSFORMER_N_HEADS = 4
        cls.TRANSFORMER_N_LAYERS = 3; cls.TRANSFORMER_DFF = 128
        cls.TRANSFORMER_DROPOUT = 0.15; cls.TRANSFORMER_LEARNING_RATE = 3e-4
        cls.VANILLA_TRANSFORMER_LR = 1e-4; cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.05
        cls.TRANSFORMER_USE_WARMUP_COSINE = True
        cls.PATCHTST_USE_REVIN = True
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True; cls.VANILLA_SEASONAL_BLEND_INIT = 0.40
        cls.VANILLA_HUBER_DELTA = 0.05
        cls.XGB_N_ESTIMATORS = 500; cls.XGB_COLSAMPLE = 0.40
        cls.GEN_AR_SIGMA = 0.030
        cls.GEN_INDUSTRIAL_LOADS = 8; cls.GEN_CITY_DISTRICTS = 12
        logging.getLogger("smart_grid").info(
            "Optimal mode: DAYS=%d HH=%d EPOCHS=%d HIST=%d | "
            "LSTM TCN=%d BiLSTM=%d heads=%d lr=%.0e | "
            "Trans d=%d h=%d L=%d lr=%.0e (VanTr %.0e) | STORAGE=%d ч",
            cls.DAYS, cls.HOUSEHOLDS, cls.EPOCHS, cls.HISTORY_LENGTH,
            cls.LSTM_TCN_FILTERS, cls.LSTM_UNITS_1, cls.LSTM_ATTN_HEADS,
            cls.LSTM_LEARNING_RATE,
            cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.TRANSFORMER_N_LAYERS,
            cls.TRANSFORMER_LEARNING_RATE, cls.VANILLA_TRANSFORMER_LR,
            cls.STORAGE_HORIZON)

    @classmethod
    def set_full_mode(cls):
        """Максимальное качество (full mode): больше данных + более ёмкие seq-модели."""
        cls.DAYS = 730; cls.HOUSEHOLDS = 2500; cls.EPOCHS = 320
        cls.PATIENCE = 35; cls.LR_PATIENCE = 12
        cls.HISTORY_LENGTH = 192; cls.STORAGE_HORIZON = 720; cls.N_FEATURES = 26
        cls.BATCH_SIZE = 8
        # На 730 днях и большом количестве окон можно использовать более ёмкий LSTM.
        cls.LSTM_UNITS_1 = 64; cls.LSTM_UNITS_2 = 64; cls.LSTM_UNITS_3 = 64
        cls.LSTM_ATTN_HEADS = 4; cls.LSTM_TCN_FILTERS = 32
        cls.DROPOUT_RATE = 0.18; cls.LSTM_LEARNING_RATE = 1.2e-4; cls.LSTM_USE_COSINE_DECAY = False
        cls.LSTM_SEASONAL_BLEND_INIT = 0.35; cls.LSTM_HUBER_DELTA = 0.05
        cls.TRANSFORMER_D_MODEL = 96; cls.TRANSFORMER_N_HEADS = 4
        cls.TRANSFORMER_N_LAYERS = 3; cls.TRANSFORMER_DFF = 192
        cls.TRANSFORMER_DROPOUT = 0.15; cls.TRANSFORMER_LEARNING_RATE = 3e-4
        cls.VANILLA_TRANSFORMER_LR = 1e-4; cls.TRANSFORMER_STOCHASTIC_DEPTH = 0.08
        cls.TRANSFORMER_USE_WARMUP_COSINE = True
        cls.PATCHTST_USE_REVIN = True
        cls.VANILLA_USE_SEASONAL_RESIDUAL = True; cls.VANILLA_SEASONAL_BLEND_INIT = 0.40
        cls.VANILLA_HUBER_DELTA = 0.05
        cls.XGB_N_ESTIMATORS = 900; cls.XGB_COLSAMPLE = 0.35
        cls.GEN_AR_SIGMA = 0.028
        cls.GEN_INDUSTRIAL_LOADS = 10; cls.GEN_CITY_DISTRICTS = 16
        logging.getLogger("smart_grid").info(
            "Full mode: DAYS=%d HH=%d EPOCHS=%d HIST=%d | "
            "LSTM BiLSTM=%d TCN=%d attn=%dh lr=%.0e drop=%.2f | "
            "Trans d=%d h=%d L=%d dff=%d lr=%.0e van_lr=%.0e | Districts=%d",
            cls.DAYS, cls.HOUSEHOLDS, cls.EPOCHS, cls.HISTORY_LENGTH,
            cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS, cls.LSTM_ATTN_HEADS, cls.LSTM_LEARNING_RATE, cls.DROPOUT_RATE,
            cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.TRANSFORMER_N_LAYERS, cls.TRANSFORMER_DFF,
            cls.TRANSFORMER_LEARNING_RATE, cls.VANILLA_TRANSFORMER_LR, cls.GEN_CITY_DISTRICTS)

    @classmethod
    def print_summary(cls):
        log = logging.getLogger("smart_grid")
        log.info("─" * 50)
        log.info("КОНФИГУРАЦИЯ:")
        log.info("  Данные:      %d дней, %d домохозяйств", cls.DAYS, cls.HOUSEHOLDS)
        log.info("  Признаки:    %d ковариат на шаг", cls.N_FEATURES)
        log.info("  История:     %d ч → прогноз %d ч", cls.HISTORY_LENGTH, cls.FORECAST_HORIZON)
        log.info("  Обучение:    %d эпох, batch=%d, patience=%d", cls.EPOCHS, cls.BATCH_SIZE, cls.PATIENCE)
        log.info("  LSTM:        BiLSTM=%d TCN=%d attn=%dh drop=%.2f lr=%g huber=%.2f input=(%d,%d)",
                 cls.LSTM_UNITS_1, cls.LSTM_TCN_FILTERS, cls.LSTM_ATTN_HEADS,
                 cls.DROPOUT_RATE, cls.LSTM_LEARNING_RATE, cls.LSTM_HUBER_DELTA,
                 cls.HISTORY_LENGTH, cls.N_FEATURES)
        log.info("  Transformer: d=%d h=%d L=%d dff=%d drop=%.2f lr=%g (Vanilla lr=%g) input=(%d,%d)",
                 cls.TRANSFORMER_D_MODEL, cls.TRANSFORMER_N_HEADS, cls.TRANSFORMER_N_LAYERS,
                 cls.TRANSFORMER_DFF, cls.TRANSFORMER_DROPOUT,
                 cls.TRANSFORMER_LEARNING_RATE, cls.VANILLA_TRANSFORMER_LR,
                 cls.HISTORY_LENGTH, cls.N_FEATURES)
        log.info("  XGBoost:     n_est=%d depth=%d col=%.2f",
                 cls.XGB_N_ESTIMATORS, cls.XGB_MAX_DEPTH, cls.XGB_COLSAMPLE)
        log.info("  Сценарий:    %s | EV=%.1f%% Solar=%.1f%% DSR=%d соб./год",
                 cls.GEN_SCENARIO, cls.GEN_EV_PENETRATION*100,
                 cls.GEN_SOLAR_PENETRATION*100, cls.GEN_DSR_EVENTS_PER_YEAR)
        log.info("  Нагрузка:    %.0f кВт·ч/мес на домохозяйство, непром. доля %.0f%%",
                 cls.GEN_KWH_PER_HOUSEHOLD_MONTH, cls.GEN_NONRESIDENTIAL_SHARE*100)
        log.info("  Климат:      среднегод %.1f °C, амплитуда ±%.1f °C (янв %.1f / июль %.1f)",
                 cls.GEN_TEMP_ANNUAL_MEAN, cls.GEN_TEMP_ANNUAL_AMPLITUDE,
                 cls.GEN_TEMP_ANNUAL_MEAN - cls.GEN_TEMP_ANNUAL_AMPLITUDE,
                 cls.GEN_TEMP_ANNUAL_MEAN + cls.GEN_TEMP_ANNUAL_AMPLITUDE)
        log.info("  Тарифы:      пик=%.2f полупик=%.2f ночь=%.2f руб/кВт·ч "
                 "(пик 07–10 и 17–21)",
                 cls.TARIFF_PEAK, cls.TARIFF_HALF_PEAK, cls.TARIFF_NIGHT)
        log.info("  Батарея:     %.0f кВт·ч, SOC %.0f%%→%.0f%% (ΔE=%.0f кВт·ч), КПД %.0f%%",
                 cls.BATTERY_CAPACITY, cls.BATTERY_MIN_SOC*100, cls.BATTERY_MAX_SOC*100,
                 (cls.BATTERY_MAX_SOC-cls.BATTERY_MIN_SOC)*cls.BATTERY_CAPACITY,
                 cls.BATTERY_EFFICIENCY*100)
        log.info("  Экономика:   CAPEX %.1f млн руб, деградация %.2f руб/кВт·ч оборота",
                 cls.BATTERY_COST_RUB/1e6, cls.BATTERY_CYCLE_COST)
        log.info("─" * 50)