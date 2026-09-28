# -*- coding: utf-8 -*-
"""
Чтение каталогов прогонов results/runs для экранов «Модели» и «Прогноз».

Только чтение: интерфейс ничего не пересчитывает и не пишет в results/.
Имя прогона из адреса принимается, только если оно проходит регулярное
выражение И есть в листинге каталога — так «..» или абсолютный путь не
открывают чужие файлы.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

RUN_RE = re.compile(r"^\d{8}_\d{6}_[\w.-]+$")

MODEL_NAMES = {
    "Naive24 (сутки)": "как вчера", "Naive24": "как вчера",
    "Naive168 (неделя)": "как неделю назад",
    "HourlyProfile": "профиль часа недели",
}


def display_model(name: str) -> str:
    return MODEL_NAMES.get(name, name)


class Runs:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)

    def path(self, name: str, *parts: str) -> str:
        if not RUN_RE.match(name or "") or not os.path.isdir(self.root) \
                or name not in os.listdir(self.root):
            raise KeyError("прогон не найден")
        return os.path.join(self.root, name, *parts)

    def meta(self, name: str) -> Optional[Dict[str, Any]]:
        path = self.path(name, "run_metadata.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def scan(self, include_smoke: bool = False) -> List[Dict[str, Any]]:
        """Прогоны с метаданными и метриками, новые сверху; неполные пропускаются."""
        if not os.path.isdir(self.root):
            return []
        rows = []
        for name in sorted(os.listdir(self.root), reverse=True):
            if not RUN_RE.match(name):
                continue
            meta = self.meta(name)
            if not meta or not os.path.exists(self.path(name, "metrics.csv")):
                continue
            mode = str(meta.get("mode", ""))
            if "smoke" in mode and not include_smoke:
                continue
            # Прогоны до появления поля status (до 28.09.2026) его не имеют:
            # показывать их «идущими» было бы неправдой.
            status = meta.get("status") or "unknown"
            rows.append({
                "name": name, "mode": mode, "dataset": meta.get("dataset") or "synthetic",
                "seed": meta.get("seed"), "status": status,
                "best": meta.get("best_model_by_val"),
                "failed": len(meta.get("failed_models") or []) + len(meta.get("failed_blocks") or []),
                "started": name[:8],
                "panel": mode.startswith("panel"),
                "has_models": os.path.exists(self.path(name, "models", "manifest.json")),
                "has_panel_series": os.path.exists(self.path(name, "forecast_series_panel_test.csv")),
                "has_dashboard": os.path.exists(self.path(name, "dashboard.html")),
            })
        return rows

    def models_table(self, name: str) -> List[Dict[str, Any]]:
        """
        Модели на тесте: ошибка и отношение к «как вчера» на тех же сутках.

        Отношение считается по MAE на тесте, а не из MASE: знаменатель MASE —
        ошибка «как вчера» на обучающем отрезке, и «точнее на N %» из него
        было бы неверным.
        """
        frame = pd.read_csv(self.path(name, "metrics.csv"), encoding="utf-8-sig")
        test = frame[frame["split"] == "test"] if "split" in frame.columns else frame
        naive = test[test["model"].isin(["Naive24 (сутки)", "Naive24"])]
        naive_mae = float(naive["MAE"].iloc[0]) if len(naive) else None
        rows = []
        for _, r in test.iterrows():
            mae = float(r["MAE"])
            rows.append({
                "model": display_model(str(r["model"])),
                "code": str(r["model"]),
                "mae": mae,
                "vs_naive": (mae / naive_mae - 1) if naive_mae else None,
                "mase": float(r["MASE"]) if "MASE" in r and pd.notna(r["MASE"]) else None,
                "train_time": float(r["train_time_sec"]) if "train_time_sec" in r and
                pd.notna(r.get("train_time_sec")) else None,
            })
        return sorted(rows, key=lambda x: x["mae"])

    def series_table(self, name: str, model: str) -> List[Dict[str, Any]]:
        path = self.path(name, "metrics_by_series.csv")
        if not os.path.exists(path):
            return []
        frame = pd.read_csv(path, encoding="utf-8-sig")
        m = frame[frame["model"] == model].set_index("series")["MAE"]
        naive = frame[frame["model"].isin(["Naive24"])].set_index("series")["MAE"]
        rows = [{"series": s, "mae": float(v),
                 "vs_naive": float(v / naive[s] - 1) if s in naive and naive[s] > 0 else None}
                for s, v in m.items()]
        return sorted(rows, key=lambda x: -(x["vs_naive"] if x["vs_naive"] is not None else 0))

    def clients_economics(self, name: str) -> List[Dict[str, Any]]:
        path = self.path(name, "economics_ru_cat4_panel", "by_client.csv")
        if not os.path.exists(path):
            return []
        frame = pd.read_csv(path, encoding="utf-8-sig")
        mpc = frame[frame["controller"] == "MPC по прогнозу"]
        bound = frame[frame["controller"] != "MPC по прогнозу"].set_index("series")
        rows = []
        for _, r in mpc.iterrows():
            rows.append({"series": r["series"], "peak_kw": float(r["peak_kw"]),
                         "annual": float(r["annual_mean"]), "lo": float(r["annual_lo"]),
                         "hi": float(r["annual_hi"]),
                         "bound": float(bound.loc[r["series"], "annual_net_savings"])
                         if r["series"] in bound.index else None,
                         "payback": float(r["payback_years"]), "n_days": int(r["n_days"])})
        return sorted(rows, key=lambda x: -x["peak_kw"])

    def panel_forecast(self, name: str, series: str) -> Optional[Dict[str, Any]]:
        """
        Проверка на истории: факт и прогноз лучшей модели на тесте прогона.

        «Как вчера» — факт, сдвинутый на сутки внутри того же файла; первая
        сутки без предыдущих в сравнение не входят.
        """
        path = self.path(name, "forecast_series_panel_test.csv")
        if not os.path.exists(path):
            return None
        frame = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["timestamp"])
        part = frame[frame["series"] == series].sort_values("timestamp")
        if part.empty:
            return None
        s = part.set_index("timestamp")
        naive = s["actual"].shift(24, freq="h").reindex(s.index)
        mask = naive.notna()
        mae_model = float((s["forecast"] - s["actual"]).abs()[mask].mean())
        mae_naive = float((naive - s["actual"]).abs()[mask].mean()) if mask.any() else None
        cols = [c for c in ("p10", "p90") if c in s.columns]
        return {
            "t": (s.index.astype("int64") // 10**9).tolist(),
            "actual": np.round(s["actual"].to_numpy(), 3).tolist(),
            "forecast": np.round(s["forecast"].to_numpy(), 3).tolist(),
            "band": {c: np.round(s[c].to_numpy(), 3).tolist() for c in cols} or None,
            "mae": mae_model, "mae_naive": mae_naive,
            "vs_naive": (mae_model / mae_naive - 1) if mae_naive else None,
            "days": int(s.index.normalize().nunique()),
            "model": display_model(str(part["model"].iloc[0])) if "model" in part else None,
        }

    def panel_series_names(self, name: str) -> List[str]:
        path = self.path(name, "forecast_series_panel_test.csv")
        if not os.path.exists(path):
            return []
        return sorted(pd.read_csv(path, encoding="utf-8-sig", usecols=["series"])["series"].unique())
