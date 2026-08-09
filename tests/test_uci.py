# -*- coding: utf-8 -*-
"""
Тесты адаптера UCI ElectricityLoadDiagrams20112014.

Проверяются свойства, нарушение которых не видно ни в одной метрике: сдвиг
суточного профиля на четверть часа, завышение масштаба вчетверо, ежегодные
выбросы в дни перевода часов, обучение на периоде до подключения клиента и
статические признаки, посчитанные с заглядыванием в тест.

Данные для тестов формируются в том же формате, что и исходный файл: точка с
запятой как разделитель, запятая как десятичный знак, метка на КОНЦЕ
интервала. Проверка на упрощённом формате не поймала бы именно те дефекты,
ради которых адаптер и написан.
"""

import numpy as np
import pandas as pd
import pytest

from data.uci import (
    UCI_URL, build_static_features, portuguese_holidays, quarter_hours_to_hourly,
    read_uci_raw, repair_dst_artifacts, select_active_series, uci_to_panel,
    _easter_sunday,
)


# ══════════════════════════════════════════════════════════════════════════════
# ФОРМИРОВАНИЕ ФАЙЛА В ФОРМАТЕ ИСХОДНОГО НАБОРА
# ══════════════════════════════════════════════════════════════════════════════
# Сам генератор файла лежит в conftest.py: им пользуется и сквозной тест
# конвейера, а два независимых описания формата разошлись бы.

@pytest.fixture(scope="module")
def uci_file(tmp_path_factory):
    from conftest import write_uci_file
    return write_uci_file(tmp_path_factory.mktemp("uci") / "LD2011_2014.txt")


# ══════════════════════════════════════════════════════════════════════════════
# ЧТЕНИЕ
# ══════════════════════════════════════════════════════════════════════════════

def test_decimal_comma_is_parsed_as_number(uci_file):
    """
    Запятая как десятичный знак разбирается в число.

    При разборе с точкой вся таблица молча стала бы объектной, и первая же
    арифметика упала бы далеко от места настоящей причины.
    """
    frame = read_uci_raw(uci_file)
    assert frame.shape[1] == 3
    assert all(np.issubdtype(t, np.floating) for t in frame.dtypes)
    assert np.isfinite(frame.to_numpy()).all()


def test_missing_file_names_the_source(tmp_path):
    """Отсутствие файла объясняется, а не падает по месту чтения."""
    with pytest.raises(FileNotFoundError, match="не скачивается автоматически"):
        read_uci_raw(str(tmp_path / "нет.txt"))
    assert UCI_URL.startswith("https://")


def test_usecols_selects_by_name_not_by_position(uci_file):
    """Колонки выбираются по имени: порядок клиентов в файле произволен."""
    frame = read_uci_raw(uci_file, usecols=["MT_003"])
    assert list(frame.columns) == ["MT_003"]

    with pytest.raises(KeyError, match="MT_999"):
        read_uci_raw(uci_file, usecols=["MT_999"])


# ══════════════════════════════════════════════════════════════════════════════
# ПЕРЕХОД К ЧАСОВОМУ ШАГУ
# ══════════════════════════════════════════════════════════════════════════════

def test_label_is_interval_end_not_start():
    """
    Отсчёт с меткой 01:00 относится к часу 00, а не 01.

    Метка описывает КОНЕЦ интервала, поэтому «01:00:00» покрывает 00:45–01:00.
    Округление метки вниз без сдвига перенесло бы этот отсчёт в следующий час
    и сместило суточный профиль у всех рядов одинаково — ошибка, невидимая в
    метриках, но разрушающая связь с календарём.
    """
    index = pd.date_range("2013-01-01 00:15:00", periods=8, freq="15min")
    values = np.zeros(8)
    values[3] = 100.0                                   # метка 01:00:00
    wide = pd.DataFrame({"MT_001": values}, index=index)

    hourly = quarter_hours_to_hourly(wide)

    assert hourly.loc[pd.Timestamp("2013-01-01 00:00"), "MT_001"] == pytest.approx(25.0)
    assert hourly.loc[pd.Timestamp("2013-01-01 01:00"), "MT_001"] == pytest.approx(0.0)


def test_power_is_converted_to_energy():
    """
    Значения — мощность в кВт; час энергии равен среднему за час.

    Простое суммирование четырёх отсчётов завысило бы масштаб вчетверо.
    """
    index = pd.date_range("2013-01-01 00:15:00", periods=4, freq="15min")
    wide = pd.DataFrame({"MT_001": [40.0, 40.0, 40.0, 40.0]}, index=index)

    hourly = quarter_hours_to_hourly(wide)
    assert hourly.iloc[0, 0] == pytest.approx(40.0), "40 кВт в течение часа = 40 кВт·ч"


# ══════════════════════════════════════════════════════════════════════════════
# ДНИ ПЕРЕВОДА ЧАСОВ
# ══════════════════════════════════════════════════════════════════════════════

def test_dst_artifacts_are_repaired_and_reported(uci_file):
    """Оба ежегодных дефекта устраняются, и каждая правка попадает в отчёт."""
    hourly = quarter_hours_to_hourly(read_uci_raw(uci_file))
    march = pd.Timestamp("2013-03-31 01:00")
    october = pd.Timestamp("2013-10-27 01:00")

    assert float(hourly.loc[march].sum()) == pytest.approx(0.0), "исходный дефект марта"

    fixed, report = repair_dst_artifacts(hourly)

    assert float(fixed.loc[march].sum()) > 0.0, "нулевой час не восстановлен"
    assert float(fixed.loc[october].sum()) < float(hourly.loc[october].sum())

    actions = {item["тип"] for item in report if "разделён" in item["действие"]
               or "заменён" in item["действие"]}
    assert actions == {"март", "октябрь"}


def test_dst_repair_skips_already_clean_data():
    """
    Правка не применяется вслепую по календарю.

    Если набор уже приведён в порядок, слепая коррекция сама внесла бы дефект:
    обнулила бы нормальный час в марте и вдвое занизила октябрьский.
    """
    index = pd.date_range("2013-03-30 00:00", "2013-11-01 00:00", freq="h")
    clean = pd.DataFrame({"MT_001": np.full(len(index), 100.0)}, index=index)

    fixed, report = repair_dst_artifacts(clean)

    assert np.allclose(fixed.to_numpy(), clean.to_numpy())
    # Оба дня перевода обязаны быть рассмотрены и отклонены: пустой отчёт
    # означал бы, что проверка прошла впустую.
    assert {item["тип"] for item in report} == {"март", "октябрь"}
    assert all("пропущена" in item["действие"] for item in report)


# ══════════════════════════════════════════════════════════════════════════════
# ОТБОР РЯДОВ
# ══════════════════════════════════════════════════════════════════════════════

def test_late_joining_client_is_excluded(uci_file):
    """
    Клиент, подключённый в середине окна, отбрасывается целиком.

    Нули до подключения — это отсутствие объекта, а не нулевое потребление.
    Обучение на таком участке подгоняет модель под несуществующие данные.
    """
    hourly = quarter_hours_to_hourly(read_uci_raw(uci_file))
    window, rejected = select_active_series(hourly, "2013-01-01", "2013-12-31")

    assert "MT_003" not in window.columns
    assert any(item["клиент"] == "MT_003" for item in rejected)
    assert set(window.columns) == {"MT_001", "MT_002"}


def test_same_client_is_kept_when_window_starts_after_connection(uci_file):
    """Тот же клиент проходит отбор, если окно начинается после подключения."""
    hourly = quarter_hours_to_hourly(read_uci_raw(uci_file))
    window, _ = select_active_series(hourly, "2013-08-01", "2013-12-31")
    assert "MT_003" in window.columns


def test_series_limit_keeps_largest_not_first(uci_file):
    """
    Ограничение числа рядов оставляет крупнейшие, а не первые по алфавиту.

    Номер клиента в наборе произволен, поэтому срез «первых N» дал бы панель
    случайного состава.
    """
    hourly = quarter_hours_to_hourly(read_uci_raw(uci_file))
    window, _ = select_active_series(hourly, "2013-08-01", "2013-12-31", max_series=1)
    assert list(window.columns) == ["MT_001"], "оставлен не крупнейший ряд"


def test_empty_selection_is_an_error(uci_file):
    """Пустая панель — отказ, а не молча возвращённая пустая таблица."""
    hourly = quarter_hours_to_hourly(read_uci_raw(uci_file))
    with pytest.raises(ValueError, match="Ни один ряд не активен"):
        select_active_series(hourly, "2013-01-01", "2013-12-31",
                             min_nonzero_share=1.0 + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# СТАТИЧЕСКИЕ ПРИЗНАКИ
# ══════════════════════════════════════════════════════════════════════════════

def test_static_features_ignore_the_test_period():
    """
    Постоянные характеристики считаются только по обучающему отрезку.

    Средний уровень или форма профиля, посчитанные по всему ряду, переносят в
    обучение сведения о тесте. Утечка такого рода не нарушает ни одной
    проверки размерностей и видна только по завышенному качеству.
    """
    index = pd.date_range("2013-01-01", periods=24 * 60, freq="h")
    values = 100.0 + 10.0 * np.sin(2 * np.pi * index.hour / 24.0)
    series = pd.Series(values, index=index)
    train_end = int(len(series) * 0.7)

    before = build_static_features(series, train_end)

    altered = series.copy()
    altered.iloc[train_end:] *= 50.0                    # тест изменён до неузнаваемости
    after = build_static_features(altered, train_end)

    assert before == after, "признаки отреагировали на изменение тестового периода"


def test_static_features_distinguish_series_shapes():
    """Признаки различают ряды с разной формой профиля, а не только с разным уровнем."""
    index = pd.date_range("2013-01-01", periods=24 * 40, freq="h")
    day = pd.Series(100.0 + 50.0 * np.sin(2 * np.pi * (index.hour - 6) / 24.0), index=index)
    flat = pd.Series(np.full(len(index), 100.0), index=index)

    a = build_static_features(day, len(index))
    b = build_static_features(flat, len(index))

    assert a["static_cv"] > b["static_cv"]
    assert a["static_load_factor"] < b["static_load_factor"]


# ══════════════════════════════════════════════════════════════════════════════
# ПРАЗДНИКИ
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("year,expected", [
    (2011, "2011-04-24"), (2012, "2012-04-08"),
    (2013, "2013-03-31"), (2014, "2014-04-20"),
])
def test_easter_matches_known_dates(year, expected):
    assert _easter_sunday(year) == pd.Timestamp(expected)


def test_holiday_calendar_is_portuguese_not_russian():
    """
    Календарь соответствует стране наблюдений.

    Российские праздники пометили бы обычные рабочие сутки как выходные и
    внесли систематическую ошибку в признак, который прямо влияет на прогноз.
    """
    days = portuguese_holidays([2013])

    assert pd.Timestamp("2013-04-25").date() in days, "День свободы — нерабочий"
    assert pd.Timestamp("2013-06-10").date() in days, "День Португалии — нерабочий"
    assert pd.Timestamp("2013-03-29").date() in days, "Страстная пятница — нерабочая"
    assert pd.Timestamp("2013-01-07").date() not in days, "российский выходной попал в набор"
    assert pd.Timestamp("2013-05-09").date() not in days


# ══════════════════════════════════════════════════════════════════════════════
# СБОРКА ПАНЕЛИ И СОВМЕСТИМОСТЬ С КОНВЕЙЕРОМ
# ══════════════════════════════════════════════════════════════════════════════

def test_panel_matches_the_pipeline_contract(uci_file):
    """
    Реальные данные проходят тот же конвейер, что и синтетические.

    Расхождение в обработке сделало бы результаты на двух наборах
    несравнимыми, а сравнение — единственная причина брать внешний набор.
    """
    df, specs, report = uci_to_panel(uci_file, start="2013-08-01", end="2013-12-31",
                                     max_series=3)

    required = {"timestamp", "city_id", "feeder_id", "consumption",
                "hour", "weekday", "is_weekend", "is_holiday", "day_of_year"}
    assert required <= set(df.columns)
    assert not any(c in df.columns for c in ("temperature", "humidity", "cloud_cover")), \
        "погода отсутствует в наборе и не должна подставляться синтетической"

    assert len(specs) == len(df.groupby(["city_id", "feeder_id"]))
    assert report["рядов отобрано"] == len(specs)

    # Все ряды на общей временной сетке: конвейер делит выборку по времени
    # одинаково для всей панели.
    lengths = df.groupby("feeder_id").size().unique()
    assert len(lengths) == 1, "ряды разной длины ломают общее разбиение"


def test_prepared_windows_have_consumption_in_channel_zero(uci_file):
    """Сквозная проверка: данные UCI доходят до окон с правильным контрактом."""
    from data.panel_preprocessing import prepare_panel_data

    df, specs, _ = uci_to_panel(uci_file, start="2013-08-01", end="2013-12-31",
                                max_series=3)
    data = prepare_panel_data(df, specs, history_length=48, forecast_horizon=24, seed=0)

    assert data["feature_names_hist"][0] == "consumption"
    assert data["consumption_channel"] == 0
    assert len(data["static_names"]) == 7
    assert data["Y_train"].shape[1] == 24
    assert np.isfinite(data["X_hist_train"]).all()


def test_real_series_scales_differ_by_orders(uci_file):
    """
    Ряды реального набора разномасштабны — ради этого он и берётся.

    На сбалансированной синтетике фидеры сопоставимы по размеру, и метрики,
    чувствительные к масштабу, там ничего не выявляют.
    """
    _, specs, report = uci_to_panel(uci_file, start="2013-08-01", end="2013-12-31",
                                    max_series=3)
    assert report["разброс средних, раз"] > 10.0
