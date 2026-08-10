# -*- coding: utf-8 -*-
"""
utils/reporting.py — Экспорт результатов в машиночитаемом виде.

Пайплайн пишет метрики не только в лог, но и в CSV/JSON. Это нужно, чтобы
таблицы пояснительной записки собирались из файла, а не переносились глазами из
`run.log`: при любом изменении конфигурации цифры в тексте иначе рассинхронятся
с фактическими результатами.

Формируемые файлы (в results/):
    metrics.csv              сводная таблица моделей (одна строка = модель × сид)
    metrics.json             то же плюс метаданные прогона
    metrics_by_horizon.csv   ошибка по каждому шагу горизонта
    diebold_mariano.csv      попарные тесты значимости различий
    storage_forecast_value.csv  экономика накопителя по источникам прогноза
    markdown_tables.md       готовые к вставке таблицы в формате Markdown
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("smart_grid.utils.reporting")

# Версия формата выходных файлов. Увеличивается при изменении состава колонок
# или структуры метаданных, чтобы старые прогоны нельзя было принять за новые.
SCHEMA_VERSION = 2

# Порядок колонок в сводной таблице: сначала то, что идёт в записку.
_METRIC_ORDER = ["model", "mode", "scenario", "seed", "split", "MAE", "RMSE", "MAPE", "sMAPE", "R2",
                 "MASE", "DW", "ACF_24", "ACF_168", "kurtosis",
                 "n_params", "train_time_sec"]


def json_safe(value: Any) -> Any:
    """
    Приводит значение к типам, допустимым в строгом JSON.

    Стандартный json.dump по умолчанию пишет NaN, Infinity и -Infinity —
    это расширение Python, а не JSON: такой файл отвергается строгими
    парсерами (в том числе `json.loads(..., parse_constant=...)`, JavaScript,
    jq). Нечисловые значения заменяются на null.

    Отдельно обрабатываются типы numpy: np.bool_ обязан стать настоящим
    boolean, иначе при сериализации через default=str он превращается в
    строку "True", и потребитель файла получает истинное значение там, где
    ожидал флаг.
    """
    import math

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]

    # bool проверяется до int: np.bool_ и bool — подтипы целых.
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if value is None or isinstance(value, (int, str)):
        return value
    return str(value)


def _reject_non_finite(_const: str):
    """Обработчик для json.loads: делает NaN/Infinity ошибкой разбора."""
    raise ValueError(f"JSON содержит недопустимое значение {_const}")


def dump_strict_json(payload: Any, path: str) -> str:
    """
    Записывает JSON без NaN/Infinity и сразу проверяет результат разбором.

    Проверка обязательна: без неё недопустимое значение обнаружится только у
    потребителя файла — например, при построении таблиц для пояснительной
    записки.
    """
    safe = json_safe(payload)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe, f, indent=2, ensure_ascii=False, allow_nan=False)

    with open(path, encoding="utf-8") as f:
        json.load(f, parse_constant=_reject_non_finite)
    return path


def collect_environment() -> Dict[str, Any]:
    """
    Фиксирует окружение прогона: версии библиотек и состояние репозитория.

    Без этого воспроизвести результат через полгода невозможно: расхождение
    в версии TensorFlow или незакоммиченная правка меняют числа, а понять
    причину постфактум уже нельзя.
    """
    import platform
    import subprocess

    def _pkg_version(name: str) -> str:
        try:
            module = __import__(name)
            return str(getattr(module, "__version__", "неизвестно"))
        except Exception:
            return "не установлен"

    def _git(*args: str) -> str:
        try:
            out = subprocess.run(["git", *args], capture_output=True, text=True,
                                 timeout=10, encoding="utf-8", errors="replace")
            return out.stdout.strip() if out.returncode == 0 else "недоступно"
        except Exception:
            return "недоступно"

    commit = _git("rev-parse", "HEAD")
    dirty = _git("status", "--porcelain")

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _pkg_version(name) for name in
                     ("numpy", "pandas", "sklearn", "tensorflow", "xgboost",
                      "statsmodels", "scipy")},
        "git_commit": commit,
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # Незакоммиченные правки означают, что коммит не описывает прогон
        # полностью — это важно знать при разборе расхождений.
        "git_dirty": bool(dirty) and dirty != "недоступно",
    }


def make_run_dir(output_dir: str, mode: str, scenario: str, seed: int) -> str:
    """
    Создаёт отдельный каталог для результатов прогона.

    Каждый запуск пишет в собственную директорию, поэтому результаты разных
    режимов и сценариев не перезаписывают друг друга и не смешиваются:
    сравнивать smoke-прогон с optimal по одному и тому же файлу нельзя.
    """
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{mode}_{scenario}_seed{seed}"
    path = os.path.join(output_dir, "runs", name)
    os.makedirs(path, exist_ok=True)

    # Отметка «выполняется» ставится сразу. Прогон, упавший до записи
    # метаданных, оставлял каталог с одними подпапками, неотличимый от
    # прерванного вручную или от успешного, у которого файлы просто не
    # прочитались: единственным признаком было отсутствие run_metadata.json.
    write_run_metadata(path, {"mode": mode, "scenario": scenario, "seed": seed,
                              "status": "running", "started_at": stamp})
    logger.info("Каталог прогона: %s", path)
    return path


def write_run_metadata(run_dir: str, meta: Dict[str, Any]) -> str:
    """
    Сохраняет метаданные прогона рядом с его результатами.

    Поле status по умолчанию «completed»: функция вызывается в конце успешного
    прогона. Отметку «running» ставит make_run_dir, и она перезаписывается
    здесь — так незавершённый прогон остаётся видимым по своему каталогу.
    """
    os.makedirs(run_dir, exist_ok=True)
    payload = dict(meta)
    payload.setdefault("status", "completed")
    payload["schema_version"] = SCHEMA_VERSION
    payload["environment"] = collect_environment()
    path = os.path.join(run_dir, "run_metadata.json")
    dump_strict_json(payload, path)
    logger.info("Метаданные прогона: %s", path)
    return path


def _ordered_frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    cols = [c for c in _METRIC_ORDER if c in df.columns]
    cols += [c for c in df.columns if c not in cols]
    return df[cols]


def export_metrics(
    all_metrics: Dict[str, Dict[str, float]],
    output_dir: str,
    seed: int,
    split: str = "test",
    run_meta: Optional[Dict[str, Any]] = None,
    append: bool = True,
    mode: str = "",
    scenario: str = "",
) -> str:
    """
    Сохраняет сводную таблицу метрик в CSV и JSON.

    Parameters
    ----------
    all_metrics : {имя модели: словарь метрик}
    append : bool
        Дописывать к существующему CSV. Нужно для прогонов на нескольких сидах:
        каждый прогон добавляет свои строки, а агрегация делается по колонке seed.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Режим и сценарий обязаны попасть в таблицу: без них строки прогонов
    # smoke, fast и optimal с одним сидом неразличимы, затирают друг друга при
    # дозаписи, а агрегация по сидам смешивает несопоставимые результаты.
    mode = mode or (run_meta or {}).get("mode", "")
    scenario = scenario or (run_meta or {}).get("scenario", "")

    rows = []
    for name, m in all_metrics.items():
        row: Dict[str, Any] = {"model": name, "mode": mode, "scenario": scenario,
                               "seed": seed, "split": split}
        row.update({k: v for k, v in m.items()})
        rows.append(row)

    df_new = _ordered_frame(rows)
    csv_path = os.path.join(output_dir, "metrics.csv")

    if append and os.path.exists(csv_path):
        try:
            df_old = pd.read_csv(csv_path)
            # Перезаписывается только та же комбинация режим+сценарий+сид+сплит.
            same = (df_old.get("seed") == seed) & (df_old.get("split") == split)
            if "mode" in df_old.columns:
                same &= df_old["mode"].fillna("") == mode
            if "scenario" in df_old.columns:
                same &= df_old["scenario"].fillna("") == scenario
            mask = ~same
            df_new = pd.concat([df_old[mask], df_new], ignore_index=True)
        except Exception as exc:
            logger.warning("Не удалось прочитать существующий metrics.csv (%s), "
                           "файл будет перезаписан.", exc)

    df_new.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = os.path.join(output_dir, "metrics.json")
    dump_strict_json({"run": run_meta or {}, "metrics": rows}, json_path)

    logger.info("Метрики сохранены: %s и %s", csv_path, json_path)
    return csv_path


def export_horizon_metrics(
    per_horizon: Dict[str, Dict[str, List[float]]],
    output_dir: str,
    seed: int,
) -> str:
    """Сохраняет метрики по каждому шагу горизонта (длинный формат)."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for model, m in per_horizon.items():
        for i, h in enumerate(m["h"]):
            row = {"model": model, "seed": seed, "horizon_step": h}
            for key in ("MAE", "RMSE", "MAPE", "R2", "MASE"):
                if key in m:
                    row[key] = m[key][i]
            rows.append(row)

    path = os.path.join(output_dir, "metrics_by_horizon.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Метрики по горизонтам сохранены: %s", path)
    return path


def export_dm_tests(
    dm_rows: List[Dict[str, Any]],
    output_dir: str,
) -> str:
    """Сохраняет результаты попарных тестов Диболда–Мариано."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "diebold_mariano.csv")
    pd.DataFrame(dm_rows).to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Тесты Диболда–Мариано сохранены: %s", path)
    return path


def export_storage_results(
    storage_results: Dict[str, Any],
    output_dir: str,
    actual: Optional[np.ndarray] = None,
    forecasts: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """Сохраняет экономику накопителя по источникам прогноза."""
    os.makedirs(output_dir, exist_ok=True)
    oracle = storage_results.get("Идеальный прогноз")
    oracle_savings = oracle.net_savings if oracle is not None else None

    rows = []
    for name, res in storage_results.items():
        mae = None
        if actual is not None and forecasts is not None and name in forecasts:
            mae = float(np.mean(np.abs(np.asarray(actual) - np.asarray(forecasts[name]))))
        rows.append({
            "forecast_source": name,
            "forecast_MAE": mae,
            "policy": res.policy,
            "baseline_cost": round(res.baseline_cost, 2),
            "optimized_cost": round(res.optimized_cost, 2),
            "gross_savings": round(res.gross_savings, 2),
            "degradation_cost": round(res.degradation_cost, 2),
            "om_cost": round(res.om_cost, 2),
            "net_savings": round(res.net_savings, 2),
            "net_savings_pct": round(res.net_savings_pct, 4),
            "shortfall_vs_oracle": (round(oracle_savings - res.net_savings, 2)
                                    if oracle_savings is not None else None),
            "peak_before_kw": round(res.peak_before_kw, 1),
            "peak_after_kw": round(res.peak_after_kw, 1),
            "payback_years": round(res.payback_years, 2),
            "energy_cycled_kwh": round(res.total_energy_cycled, 1),
        })

    path = os.path.join(output_dir, "storage_forecast_value.csv")
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Экономика накопителя сохранена: %s", path)
    return path


def export_markdown_tables(
    all_metrics: Dict[str, Dict[str, float]],
    output_dir: str,
    dm_rows: Optional[List[Dict[str, Any]]] = None,
    storage_results: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Формирует готовые к вставке в пояснительную записку таблицы Markdown.

    Так цифры в тексте гарантированно совпадают с результатами последнего
    прогона: таблица копируется целиком, а не набирается вручную.
    """
    os.makedirs(output_dir, exist_ok=True)
    lines: List[str] = []

    lines.append("# Таблицы результатов (сгенерированы автоматически)\n")
    lines.append("## Таблица 1. Сравнение моделей на тестовой выборке\n")
    lines.append("| Модель | MAE, кВт·ч | RMSE | MAPE, % | R² | MASE | Параметры | Обучение, с |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in sorted(all_metrics, key=lambda n: all_metrics[n].get("MAE", 9e18)):
        m = all_metrics[name]
        params = int(m.get("n_params", 0))
        lines.append(
            f"| {name} | {m.get('MAE', float('nan')):.1f} | {m.get('RMSE', float('nan')):.1f} "
            f"| {m.get('MAPE', float('nan')):.2f} | {m.get('R2', float('nan')):.4f} "
            f"| {m.get('MASE', float('nan')):.3f} "
            f"| {params if params else '—'} | {m.get('train_time_sec', 0):.1f} |"
        )
    lines.append("\n*MASE < 1 означает превосходство над сезонно-наивным прогнозом.*\n")

    if dm_rows:
        lines.append("## Таблица 2. Тест Диболда–Мариано (значимость различий)\n")
        lines.append("| Модель A | Модель B | DM | p-value | Вывод |")
        lines.append("|---|---|---|---|---|")
        for r in dm_rows:
            lines.append(f"| {r['model_a']} | {r['model_b']} | {r['DM']:.3f} "
                         f"| {r['p_value']:.4f} | {r['better']} |")
        lines.append("")

    if storage_results:
        lines.append("## Таблица 3. Экономический эффект накопителя\n")
        lines.append("| Источник прогноза | Чистая экономия, руб | Недобор к идеалу, руб | Окупаемость, лет |")
        lines.append("|---|---|---|---|")
        oracle = storage_results.get("Идеальный прогноз")
        oracle_val = oracle.net_savings if oracle is not None else None
        for name, res in sorted(storage_results.items(), key=lambda kv: -kv[1].net_savings):
            short = f"{oracle_val - res.net_savings:,.0f}".replace(",", " ") if oracle_val else "—"
            payback = f"{res.payback_years:.1f}" if res.payback_years < 999 else "∞"
            lines.append(f"| {name} | {res.net_savings:,.0f}".replace(",", " ")
                         + f" | {short} | {payback} |")
        lines.append("")

    path = os.path.join(output_dir, "markdown_tables.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Markdown-таблицы для записки: %s", path)
    return path


def aggregate_seeds(output_dir: str) -> Optional[str]:
    """
    Агрегирует metrics.csv по сидам: mean ± std для каждой модели.

    Без этого сравнение моделей опирается на один случайный прогон, и разница
    в 2–3% ничем не подкреплена.
    """
    csv_path = os.path.join(output_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if "seed" not in df.columns or df["seed"].nunique() < 2:
        logger.info("Агрегация по сидам пропущена: в metrics.csv один сид. "
                    "Запустите пайплайн с --seed 0,1,2 для оценки разброса.")
        return None

    # Агрегируются только сопоставимые прогоны: смешивать режимы нельзя.
    if "mode" in df.columns and df["mode"].nunique() > 1:
        latest = df.sort_index().iloc[-1]["mode"]
        logger.info("В metrics.csv несколько режимов — агрегация только по '%s'.", latest)
        df = df[df["mode"] == latest]
    df = df[df.get("split", "test") == "test"] if "split" in df.columns else df
    if df["seed"].nunique() < 2:
        logger.info("Агрегация по сидам пропущена: в выбранном режиме один сид.")
        return None

    num_cols = [c for c in ("MAE", "RMSE", "MAPE", "sMAPE", "R2", "MASE")
                if c in df.columns]
    agg = df.groupby("model")[num_cols].agg(["mean", "std", "count"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.sort_values("MAE_mean")

    # Режим и перечень сидов записываются в сам файл: без них таблица
    # неинтерпретируема — MAE агрегатного ряда города и MAE отдельного фидера
    # различаются в разы, и по одним числам не понять, что именно усреднено.
    agg.insert(0, "mode", str(df["mode"].iloc[-1]) if "mode" in df.columns else "")
    agg.insert(1, "seeds", ",".join(str(s) for s in sorted(df["seed"].unique())))

    path = os.path.join(output_dir, "metrics_by_seed.csv")
    agg.to_csv(path, encoding="utf-8-sig")

    logger.info("─" * 78)
    logger.info("АГРЕГАЦИЯ ПО СИДАМ (n=%d)", int(df["seed"].nunique()))
    logger.info("%-22s %18s %18s", "Модель", "MAE (mean±std)", "MASE (mean±std)")
    logger.info("─" * 78)
    for model, row in agg.iterrows():
        mase = (f"{row.get('MASE_mean', float('nan')):.3f}±{row.get('MASE_std', 0):.3f}"
                if "MASE_mean" in row else "н/д")
        logger.info("%-22s %10.1f±%-7.1f %18s",
                    model, row["MAE_mean"], row.get("MAE_std", 0), mase)
    logger.info("─" * 78)
    logger.info("Сводка по сидам сохранена: %s", path)
    return path
