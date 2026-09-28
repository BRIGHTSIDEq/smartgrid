// Поведение страниц. Без фреймворков: формы, опрос фоновых расчётов, шкала
// суток, раскрывающиеся строки и графики.
(function () {
  "use strict";
  var charts = {};

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }

  // ── Текущая выгрузка в навигации ──────────────────────────────────────────
  // «Прогноз» и «Модели» не привязаны к выгрузке; чтобы возврат в «Данные» и
  // «Экономию» не терял её, последняя выгрузка запоминается в браузере.
  // Хранилище может быть недоступно (частный режим) — тогда просто без этого.
  try {
    var here = new URLSearchParams(location.search).get("u");
    if (here) localStorage.setItem("sg_upload", here);
    var remembered = here || localStorage.getItem("sg_upload");
    if (remembered && /^[0-9a-f]{32}$/.test(remembered)) {
      $$('.tabs a[href="/data"], .tabs a[href="/economics"]').forEach(function (a) {
        a.href = a.getAttribute("href") + "?u=" + remembered;
      });
    }
  } catch (e) { /* без запоминания */ }

  // ── Автоотправка выбора ───────────────────────────────────────────────────
  $$("[data-autosubmit]").forEach(function (input) {
    input.addEventListener("change", function () { input.form.submit(); });
  });

  // ── Раскрывающиеся строки отчёта о качестве и график прибора ─────────────
  function loadMeter(series) {
    var host = $("#meter-chart");
    if (!host) return;
    var title = $("#meter-title");
    if (title) title.textContent = series;
    fetch("/data/meter.json?u=" + encodeURIComponent(host.dataset.u) +
          "&series=" + encodeURIComponent(series))
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var chart = new SGChart.TimeChart(host, {
          t: d.t, unit: "кВт·ч", days: currentDays("meter-chart", 30), problems: d.problems,
          series: [{ name: series, tipName: "расход, кВт·ч", values: d.y, color: "var(--ink)", width: 1.25 }]
        });
        charts["meter-chart"] = chart;
        var note = $("#meter-note");
        // Прибор с замечаниями открывается на участке с проблемой: иначе
        // пользователь видит последние чистые недели и не понимает, что не так.
        if (d.problems.length) {
          chart.focus(d.problems[0], 7);
          $$('[data-range-for="meter-chart"] button').forEach(function (b) {
            b.setAttribute("aria-pressed", b.dataset.days === "7" ? "true" : "false");
          });
          if (note) note.textContent = "Показан участок с первой проблемой: " + fmt.dateFull(d.problems[0]) +
            ". Проблемных часов всего: " + fmt.num(d.problems.length) + ".";
        } else if (note) {
          note.textContent = "Проблем в данных прибора не найдено.";
        }
      });
  }
  $$("tr.expandable").forEach(function (row) {
    var toggle = function () {
      var details = row.nextElementSibling, open = row.classList.toggle("open");
      if (details && details.classList.contains("details")) details.hidden = !open;
      $$("tr.expandable").forEach(function (r) { r.classList.remove("selected"); });
      row.classList.add("selected");
      loadMeter(row.dataset.series);
    };
    row.addEventListener("click", toggle);
    row.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); } });
  });
  if ($("#meter-chart")) loadMeter($("#meter-chart").dataset.series);

  // ── Кнопки периода «7 суток / 30 суток / всё» ─────────────────────────────
  function currentDays(chartId, fallback) {
    var group = $('[data-range-for="' + chartId + '"] [aria-pressed="true"]');
    return group ? Number(group.dataset.days) : fallback;
  }
  document.addEventListener("click", function (e) {
    var btn = e.target.closest(".segmented button");
    if (!btn) return;
    var group = btn.parentElement, id = group.dataset.rangeFor;
    $$("button", group).forEach(function (b) { b.setAttribute("aria-pressed", b === btn ? "true" : "false"); });
    var days = Number(btn.dataset.days);
    (group.dataset.rangeFor === "schedule-chart" ? ["schedule-chart", "soc-chart"] : [id]).forEach(function (c) {
      if (charts[c]) charts[c].setDays(days);
    });
  });

  // ── Графики, данные которых пришли в разметке ─────────────────────────────
  function initCharts(root) {
    $$("script.chart-data", root).forEach(function (node) {
      var data = JSON.parse(node.textContent), kind = node.dataset.kind;
      if (kind === "schedule") {
        charts["schedule-chart"] = new SGChart.TimeChart($("#schedule-chart", root), {
          t: data.t, unit: "кВт·ч", window: data.window, days: 7,
          series: [
            { name: "нагрузка", label: "нагрузка", values: data.load, color: "#9AA1A6", width: 1.25 },
            { name: "из сети", label: "из сети", values: data.grid, color: "var(--ink)", width: 1.5 }
          ]
        });
        charts["soc-chart"] = new SGChart.TimeChart($("#soc-chart", root), {
          t: data.t, unit: "заряд накопителя, кВт·ч", window: data.window, days: 7,
          series: [{ name: "заряд", label: "заряд", values: data.soc, color: "var(--ok)", width: 1.5 }]
        });
      } else if (kind === "monthly") {
        var host = $("#monthly-chart", root);
        SGChart.BarChart(host, {
          unit: "₽", values: data.map(function (m) { return m.gross_savings; }),
          partial: data.map(function (m) { return m.partial; }),
          labels: data.map(function (m) {
            var p = m.month.split("-");
            return fmt.monthOnly(Date.UTC(+p[0], +p[1] - 1, 1) / 1000) + (m.partial ? "*" : "");
          })
        });
      } else if (kind === "forecast") {
        var series = [
          { name: "факт", label: "факт", values: data.actual, color: "var(--ink)", width: 1.5 },
          { name: "прогноз", label: "прогноз", values: data.forecast, color: "var(--accent)", width: 2 }
        ];
        var band = data.band && data.band.p10 ? { lo: data.band.p10, hi: data.band.p90 } : null;
        charts["forecast-chart"] = new SGChart.TimeChart($("#forecast-chart", root), {
          t: data.t, unit: "кВт·ч", days: 7, series: series, band: band
        });
      }
    });
    initSpreads(root);
  }

  function initSpreads(root) {
    var groups = {};
    $$("[data-spread]", root).forEach(function (s) {
      var key = s.dataset.shared || ("own" + Math.random());
      (groups[key] = groups[key] || []).push(s);
    });
    Object.keys(groups).forEach(function (key) {
      var items = groups[key], lo = Infinity, hi = -Infinity;
      items.forEach(function (s) { lo = Math.min(lo, +s.dataset.lo); hi = Math.max(hi, +s.dataset.hi); });
      if (lo < 0) hi = Math.max(hi, 0);          // ноль на шкале — только при отрицательных
      var pad = (hi - lo) * 0.06 || 1;
      items.forEach(function (s) {
        SGChart.Spread(s, +s.dataset.lo, +s.dataset.hi, +s.dataset.mid,
                       items.length > 1 || key.indexOf("own") !== 0 ? [lo - pad, hi + pad] : null,
                       s.dataset.zero === "1");
      });
    });
  }
  initCharts(document);

  // ── Фоновые расчёты: запуск, опрос, остановка ─────────────────────────────
  function runJob(url, body, target, onDone) {
    var started = Date.now(), jobId = null;
    var progress = document.createElement("div");
    progress.className = "progress";
    progress.innerHTML = '<span data-stage>Считаем…</span><button type="button" class="button">Остановить</button>';
    target.prepend(progress);
    $("button", progress).addEventListener("click", function () {
      if (jobId) fetch("/jobs/" + jobId + "/stop", { method: "POST" });
      $("button", progress).disabled = true;
    });
    fetch(url, { method: "POST", body: body })
      .then(function (r) {
        if (!r.ok) return r.json().then(function (e) { throw new Error(e.detail || "Ошибка запроса"); });
        return r.json();
      })
      .then(function (d) { jobId = d.job; poll(); })
      .catch(function (err) { progress.outerHTML = errorHtml(err.message); });

    function poll() {
      fetch("/jobs/" + jobId).then(function (r) { return r.json(); }).then(function (s) {
        if (s.status === "running") {
          var secs = Math.round((Date.now() - started) / 1000);
          $("[data-stage]", progress).textContent = "Считаем: " + (s.stage || "подготовка") +
            " · прошло " + Math.floor(secs / 60) + ":" + ("0" + secs % 60).slice(-2);
          progress.style.setProperty("--p", Math.round(s.progress * 100) + "%");
          setTimeout(poll, 800);
          return;
        }
        progress.remove();
        onDone(s.html);
      });
    }
  }

  function errorHtml(text) {
    var d = document.createElement("div");
    d.className = "error-box";
    d.setAttribute("role", "alert");
    d.textContent = text;
    return d.outerHTML;
  }

  var econForm = $("#econ-form");
  if (econForm) {
    econForm.addEventListener("submit", function (e) {
      e.preventDefault();
      var target = $("#econ-result"), button = $('button[type="submit"]', econForm);
      button.disabled = true;
      runJob(econForm.dataset.run, new FormData(econForm), target, function (html) {
        button.disabled = false;
        target.innerHTML = html;
        initCharts(target);
      });
    });
  }

  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-whatif]");
    if (!btn) return;
    var body = $("#whatif-body");
    btn.disabled = true;
    runJob(btn.dataset.whatif, null, body, function (html) {
      body.innerHTML = html;
      initSpreads(body);
    });
  });

  // ── Поля ставок: категория и пометка «пример» ─────────────────────────────
  var category = $("[data-category]");
  function applyCategory() {
    if (!category) return;
    var c = Number(category.value), visible = {
      all: true, "two-rate": c === 4 || c === 6, "one-rate": c === 3 || c === 5, plan: c === 5 || c === 6
    };
    $$("label[data-group]").forEach(function (l) { l.hidden = !visible[l.dataset.group]; });
  }
  if (category) { category.addEventListener("change", applyCategory); applyCategory(); }

  var confirmed = document.createElement("input");
  confirmed.type = "hidden"; confirmed.name = "rates_confirmed"; confirmed.value = "";
  if (econForm) econForm.appendChild(confirmed);
  $$("input[data-example]").forEach(function (input) {
    var label = input.closest("label");
    label.classList.add("is-example");
    input.addEventListener("input", function () { label.classList.remove("is-example"); });
  });
  var confirmBtn = $("[data-confirm-rates]");
  if (confirmBtn) confirmBtn.addEventListener("click", function () {
    $$("label.is-example").forEach(function (l) { l.classList.remove("is-example"); });
    confirmed.value = "1";
    var banner = $("#example-banner");
    if (banner) banner.hidden = true;
    updateSummaries();
  });

  // ── Сводки в заголовках групп параметров ──────────────────────────────────
  function updateSummaries() {
    if (!econForm) return;
    var total = $$('input[name="meters"]', econForm).length;
    var picked = $$('input[name="meters"]:checked', econForm).length;
    var set = function (key, text) { var e = $('[data-summary="' + key + '"]'); if (e) e.textContent = text; };
    set("meters", "выбрано " + picked + " из " + total);
    var example = $$("label.is-example", econForm).length > 0 && confirmed.value !== "1";
    set("tariff", "категория " + (category ? category.value : "") + " · " + (example ? "ставки примерные" : "ваши ставки"));
    var hours = ($('input[name="peak_hours"]', econForm).value || "").split(",").filter(Boolean).map(Number);
    set("hours", hours.length ? hours[0] + ":00–" + (hours[hours.length - 1] + 1) + ":00" : "не выбраны");
    var share = $('input[name="power_share"]', econForm), capex = $('input[name="capex_per_kwh"]', econForm);
    set("battery", (share ? share.value : "") + " % от максимума · " + (capex ? capex.value : "") + " ₽/кВт·ч");
  }
  if (econForm) {
    econForm.addEventListener("input", updateSummaries);
    econForm.addEventListener("change", updateSummaries);
  }

  // ── Заготовки часов пика ──────────────────────────────────────────────────
  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-hours]");
    if (!btn) return;
    var hours = [];
    btn.dataset.hours.split(",").forEach(function (range) {
      var p = range.split("-").map(Number);
      for (var h = p[0]; h < p[1]; h++) hours.push(h);
    });
    var scale = $("[data-dayscale]");
    if (scale && scale.setHours) scale.setHours(hours);
  });

  // ── Шкала суток: плановые часы пиковой нагрузки ───────────────────────────
  $$("[data-dayscale]").forEach(function (scale) {
    var input = scale.parentElement.querySelector('input[name="peak_hours"]');
    var on = {}, dragging = false, mode = true;
    (scale.dataset.value || "").split(",").forEach(function (h) { if (h !== "") on[+h] = true; });
    for (var h = 0; h < 24; h++) {
      var cell = document.createElement("span");
      cell.dataset.hour = h;
      cell.textContent = h % 3 === 0 ? h : "";
      cell.title = (h < 10 ? "0" : "") + h + ":00–" + (h + 1 < 10 ? "0" : "") + (h + 1) + ":00";
      scale.appendChild(cell);
    }
    var axis = document.createElement("div");
    axis.className = "dayscale-axis";
    scale.after(axis);
    function sync() {
      $$("span", scale).forEach(function (c) { c.classList.toggle("on", !!on[+c.dataset.hour]); });
      var hours = Object.keys(on).filter(function (k) { return on[k]; }).map(Number).sort(function (a, b) { return a - b; });
      input.value = hours.join(",");
      axis.textContent = hours.length ? "Выделено " + hours[0] + ":00–" + (hours[hours.length - 1] + 1) +
        ":00 (" + hours.length + " ч)" : "Часы не выбраны";
      var label = scale.closest("fieldset");
      if (label) label.classList.remove("is-example");
    }
    scale.setHours = function (hours) {
      on = {}; hours.forEach(function (h) { on[h] = true; }); sync(); updateSummaries();
    };
    scale.addEventListener("pointerdown", function (e) {
      var c = e.target.closest("span"); if (!c) return;
      dragging = true; mode = !on[+c.dataset.hour]; on[+c.dataset.hour] = mode; sync();
    });
    scale.addEventListener("pointerover", function (e) {
      var c = e.target.closest("span"); if (!dragging || !c) return;
      on[+c.dataset.hour] = mode; sync();
    });
    document.addEventListener("pointerup", function () { dragging = false; });
    sync();
  });
  updateSummaries();
})();
