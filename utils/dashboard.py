# -*- coding: utf-8 -*-
"""
Автономная HTML-сводка прогона.

Один файл dashboard.html в каталоге прогона: статус и отказы, таблица
метрик, экономика по ценовой категории, экономика по клиентам и графики.
Изображения встраиваются в файл, поэтому его можно переслать заказчику или
руководителю без сервера и без Python.

    python -m utils.dashboard results/runs/<прогон>
"""

import base64
import html
import json
import os
import sys
from typing import List, Optional

import pandas as pd

_STYLE = """
:root { --bg:#fff; --fg:#1d2330; --muted:#5b6475; --line:#dfe3ea; --ok:#1f7a4d;
        --warn:#a15c00; --bad:#b3261e; --card:#f6f7f9; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171c; --fg:#e7eaf0; --muted:#9aa3b2; --line:#2c323c; --ok:#5cc99a;
          --warn:#e0a458; --bad:#ef8b83; --card:#1b1f26; }
}
body { background:var(--bg); color:var(--fg); margin:0; padding:24px 16px;
       font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1100px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 4px; } h2 { font-size:17px; margin:28px 0 8px; }
.muted { color:var(--muted); }
.status { display:inline-block; padding:2px 10px; border-radius:12px; font-weight:600;
          border:1px solid currentColor; }
.completed { color:var(--ok); } .partial { color:var(--warn); } .failed, .running { color:var(--bad); }
.table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { background:var(--card); font-weight:600; }
figure { margin:12px 0; } figure img { max-width:100%; height:auto; border:1px solid var(--line);
                                       border-radius:6px; background:#fff; }
figcaption { color:var(--muted); font-size:13px; }
"""


def _table(frame: pd.DataFrame, digits: int = 3) -> str:
    frame = frame.copy()
    for col in frame.columns:
        if pd.api.types.is_float_dtype(frame[col]):
            frame[col] = frame[col].map(lambda v: "" if pd.isna(v) else f"{v:,.{digits}f}"
                                        .replace(",", " "))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
                   for row in frame.itertuples(index=False))
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _csv(path: str) -> Optional[pd.DataFrame]:
    return pd.read_csv(path, encoding="utf-8-sig") if os.path.exists(path) else None


def build_dashboard(run_dir: str, max_plots: int = 12) -> str:
    """Собирает dashboard.html и возвращает путь к нему."""
    meta = {}
    meta_path = os.path.join(run_dir, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    status = str(meta.get("status", "running"))
    problems = (meta.get("failed_models") or []) + (meta.get("failed_blocks") or [])
    # Сводка строится до того, как обработчик выхода выставит итоговый статус:
    # при отказах прогон частичный, даже если в файле ещё записано completed.
    if status == "completed" and problems:
        status = "partial"
    title = os.path.basename(os.path.normpath(run_dir))

    parts: List[str] = [
        f"<h1>Прогон {html.escape(title)}</h1>",
        f'<p><span class="status {html.escape(status)}">{html.escape(status)}</span> '
        f'<span class="muted">режим {html.escape(str(meta.get("mode", "—")))}, '
        f'данные {html.escape(str(meta.get("dataset", "—")))}, '
        f'сид {html.escape(str(meta.get("seed", "—")))}, '
        f'лучшая по валидации: {html.escape(str(meta.get("best_model_by_val", "—")))}</span></p>',
    ]
    if problems:
        items = "".join(f"<li>{html.escape(str(p.get('model') or p.get('block')))}: "
                        f"{html.escape(str(p.get('error')))}</li>" for p in problems)
        parts.append(f"<h2>Отказы</h2><ul>{items}</ul>")
    if meta.get("error"):
        parts.append(f"<h2>Ошибка</h2><p>{html.escape(str(meta['error']))}</p>")

    metrics = _csv(os.path.join(run_dir, "metrics.csv"))
    if metrics is not None:
        cols = [c for c in ("model", "split", "MAE", "MASE", "MAE_macro", "sMAPE", "R2",
                            "train_time_sec") if c in metrics.columns]
        test = metrics[metrics["split"] == "test"] if "split" in metrics else metrics
        if "MASE" in test.columns:
            test = test.sort_values("MASE")
        parts.append("<h2>Метрики на тесте</h2>" + _table(test[cols]))

    for sub, label in (("economics_ru_cat4", "Экономика накопителя, ценовая категория 4"),):
        summary = _csv(os.path.join(run_dir, sub, "summary.csv"))
        if summary is not None:
            cols = [c for c in ("source", "forecast_MAE", "net_savings", "annual_mean",
                                "annual_lo", "annual_hi", "share_of_bound", "payback_years")
                    if c in summary.columns]
            parts.append(f"<h2>{label}</h2>" + _table(summary[cols], digits=1)
                         + '<p class="muted">Интервал годовой экономии — блочный бутстреп '
                           'по неделям. Ставки и окна часов СО примерные.</p>')
    clients = _csv(os.path.join(run_dir, "economics_ru_cat4_panel", "by_client.csv"))
    if clients is not None:
        mpc = clients[clients["controller"] == "MPC по прогнозу"]
        cols = [c for c in ("series", "peak_kw", "forecast_MAE", "annual_mean", "annual_lo",
                            "annual_hi", "payback_years") if c in mpc.columns]
        parts.append("<h2>Экономика по клиентам (MPC по прогнозу)</h2>"
                     + _table(mpc[cols].sort_values("annual_mean", ascending=False), digits=1))

    plots_dir = os.path.join(run_dir, "plots")
    if os.path.isdir(plots_dir):
        figures = []
        for name in sorted(os.listdir(plots_dir))[:max_plots]:
            if not name.lower().endswith(".png"):
                continue
            with open(os.path.join(plots_dir, name), "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            figures.append(f'<figure><img alt="{html.escape(name)}" '
                           f'src="data:image/png;base64,{data}"><figcaption>'
                           f'{html.escape(name)}</figcaption></figure>')
        if figures:
            parts.append("<h2>Графики</h2>" + "".join(figures))

    page = (f'<!doctype html><html lang="ru"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Прогон {html.escape(title)}</title><style>{_STYLE}</style></head>"
            f"<body><main>{''.join(parts)}</main></body></html>")
    path = os.path.join(run_dir, "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    return path


if __name__ == "__main__":
    print(build_dashboard(sys.argv[1]))
