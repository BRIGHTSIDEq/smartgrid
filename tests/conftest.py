import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import numpy as np
import pandas as pd
import pytest


def write_uci_file(path, start="2013-01-01 00:15:00", periods=35_040,
                   connect_date="2013-07-01"):
    """
    Создаёт файл, повторяющий устройство LD2011_2014.txt.

    Воспроизводятся все особенности, которые адаптер обязан учесть: метка на
    конце интервала, разные масштабы клиентов, позднее подключение одного из
    них и оба дефекта дней перевода часов.

    Дни перевода вычисляются по периоду данных, а не задаются датами: с
    зашитым годом фикстура на другом интервале молча теряла бы дефекты, и
    сквозные проверки перестали бы их касаться.
    """
    index = pd.date_range(start, periods=periods, freq="15min")
    starts = index - pd.Timedelta(minutes=15)
    hours = starts.hour.to_numpy()
    rng = np.random.RandomState(0)

    shape = 1.0 + 0.6 * np.sin(2 * np.pi * (hours - 6) / 24.0)
    big = 400.0 * shape * (1.0 + 0.05 * rng.normal(size=periods))
    small = 8.0 * shape * (1.0 + 0.05 * rng.normal(size=periods))
    late = 50.0 * shape * (1.0 + 0.05 * rng.normal(size=periods))
    late[starts < pd.Timestamp(connect_date)] = 0.0        # клиент ещё не подключён

    frame = pd.DataFrame({"MT_001": big, "MT_002": small, "MT_003": late},
                         index=index).clip(lower=0.0)

    for year in sorted({int(t.year) for t in starts}):
        for month, factor in ((3, 0.0), (10, 2.0)):
            days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
            days = days[days.month == month]
            switch = days[days.dayofweek == 6][-1]         # последнее воскресенье
            hour = (starts >= switch + pd.Timedelta(hours=1)) & \
                   (starts < switch + pd.Timedelta(hours=2))
            if not hour.any():
                continue
            # Март: час обнулён у всех клиентов. Октябрь: тот же час содержит
            # потребление за два часа сразу.
            frame.loc[hour, :] = frame.loc[hour, :] * factor

    frame.index.name = ""
    frame.to_csv(path, sep=";", decimal=",", float_format="%.5f")
    return str(path)


@pytest.fixture(scope="session")
def uci_file_factory():
    """
    Отдаёт функцию создания файла в формате LD2011_2014.txt.

    Общий помощник для тестов адаптера и сквозного прогона конвейера: два
    независимых описания формата разошлись бы, и проверки перестали бы
    относиться к одним и тем же данным.
    """
    return write_uci_file
