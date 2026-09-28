// Поведение страниц. Без фреймворков: формы, опрос фоновых расчётов, шкала
// суток, ковёр и графики, мнемосхема. Движение — только в ответ на действие
// и никогда при «меньше движения» (SGChart.calm).
(function () {
  "use strict";
  var charts = {};
  var calm = SGChart.calm;

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function store(kind) { try { return window[kind]; } catch (e) { return null; } }
  function get(kind, key) { try { var s = store(kind); return s ? s.getItem(key) : null; } catch (e) { return null; } }
  function put(kind, key, value) { try { var s = store(kind); if (s) s.setItem(key, value); } catch (e) { /* без запоминания */ } }

  // ── Текущая выгрузка в навигации ──────────────────────────────────────────
  // «Прогноз» и «Модели» не привязаны к выгрузке; чтобы возврат в «Данные» и
  // «Экономию» не терял её, последняя выгрузка запоминается в браузере.
  var here = new URLSearchParams(location.search).get("u");
  if (here) put("localStorage", "sg_upload", here);
  var remembered = here || get("localStorage", "sg_upload");
  if (remembered && /^[0-9a-f]{32}$/.test(remembered)) {
    $$('.tabs a[href="/data"], .tabs a[href="/economics"], .brand[href="/data"]').forEach(function (a) {
      a.href = a.getAttribute("href") + "?u=" + remembered;
    });
  }

  // ── Мнемосхема: лампа, сменившая состояние, мигает один раз ───────────────
  $$(".mnemo .node").forEach(function (node) {
    var key = "sg_lamp_" + (here || "none") + "_" + node.dataset.node, before = get("sessionStorage", key);
    if (before && before !== node.dataset.lamp) node.classList.add("changed");
    put("sessionStorage", key, node.dataset.lamp);
  });

  // ── Загрузка: выбор файла и перетаскивание ────────────────────────────────
  $$("[data-autosubmit]").forEach(function (input) {
    input.addEventListener("change", function () { if (input.files && input.files.length) input.form.submit(); });
  });
  var drop = $("#dropzone");
  if (drop) {
    var depth = 0;
    drop.addEventListener("dragenter", function (e) { e.preventDefault(); depth++; drop.classList.add("over"); });
    drop.addEventListener("dragover", function (e) { e.preventDefault(); });
    drop.addEventListener("dragleave", function () { if (--depth <= 0) { depth = 0; drop.classList.remove("over"); } });
    drop.addEventListener("drop", function (e) {
      e.preventDefault();
      drop.classList.remove("over");
      var input = $('input[type="file"]', drop);
      if (!e.dataTransfer.files.length) return;
      try { input.files = e.dataTransfer.files; } catch (err) { return; }
      drop.submit();
    });
  }

  // ── Разбор выгрузки: ответы клавишами 1–4 ─────────────────────────────────
  var parseForm = $("#parse-form");
  if (parseForm) {
    document.addEventListener("keydown", function (e) {
      if (e.target.closest("input:not([type=radio]), select, textarea") || e.ctrlKey || e.altKey || e.metaKey) return;
      var input = $('input[data-key="' + e.key + '"]', parseForm);
      if (input) { input.checked = true; input.focus({ preventScroll: true }); }
    });
  }

  // ── Отчёт о качестве: строки, ковёр и график прибора ──────────────────────
  function loadMeter(series) {
    var host = $("#meter-chart");
    if (!host) return;
    var title = $("#meter-title");
    if (title) title.textContent = series;
    fetch("/data/meter.json?u=" + encodeURIComponent(host.dataset.u) + "&series=" + encodeURIComponent(series))
      .then(function (r) { if (!r.ok) throw new Error(); return r.json(); })
      .then(function (d) {
        var chart = new SGChart.TimeChart(host, {
          t: d.t, unit: "кВт·ч", days: currentDays("meter-chart", 30), problems: d.problems,
          series: [{ name: series, tipName: "расход, кВт·ч", values: d.y, color: "var(--ink)", width: 1.25 }]
        });
        charts["meter-chart"] = chart;
        var carpet = $("#meter-carpet");
        if (carpet) {
          var canvas = $("canvas", carpet);
          delete canvas.dataset.shown;
          SGChart.Carpet(carpet, d, function (day) { chart.focus(day + 2 * 86400, 7); press("meter-chart", "7"); });
        }
        var note = $("#meter-note");
        // Прибор с замечаниями открывается на участке с проблемой: иначе
        // пользователь видит последние чистые недели и не понимает, что не так.
        if (d.problems.length) {
          chart.focus(d.problems[0], 7);
          press("meter-chart", "7");
          if (note) note.textContent = "Показан участок с первой проблемой: " + fmt.dateFull(d.problems[0]) +
            ". Проблемных часов всего: " + fmt.num(d.problems.length) + ".";
        } else if (note) {
          note.textContent = "Проблем в данных прибора не найдено.";
        }
      })
      .catch(function () { host.innerHTML = '<p class="note" style="padding:14px">Не удалось загрузить данные прибора.</p>'; });
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

  // ── Сверка со счётом ──────────────────────────────────────────────────────
  // Цифры из счёта — удобство для этого браузера, на расчёт не влияют.
  var bill = $("table.bill-check");
  if (bill) {
    var billKey = "sg_bill_" + bill.dataset.u;
    var saved = {};
    try { saved = JSON.parse(get("localStorage", billKey) || "{}"); } catch (e) { saved = {}; }
    $$("tr[data-month]", bill).forEach(function (row) {
      var input = $("input", row), out = $(".diff", row), ours = Number(row.dataset.kwh);
      function update() {
        var v = Number(String(input.value).replace(/[\s  ]/g, "").replace(",", "."));
        if (!input.value || !isFinite(v) || v <= 0) { out.textContent = "—"; out.className = "n diff"; return; }
        var diff = ours / v - 1;
        out.textContent = (diff > 0 ? "+" : diff < 0 ? fmt.MINUS : "") + fmt.num(Math.abs(diff) * 100, 1) + fmt.NBSP + "%";
        out.className = "n diff " + (Math.abs(diff) <= 0.03 ? "ok" : "bad");
      }
      if (saved[row.dataset.month]) input.value = saved[row.dataset.month];
      input.addEventListener("input", function () {
        saved[row.dataset.month] = input.value;
        put("localStorage", billKey, JSON.stringify(saved));
        update();
      });
      update();
    });
  }

  // ── Кнопки периода «7 суток / 30 суток / всё» ─────────────────────────────
  function currentDays(chartId, fallback) {
    var pressed = $('[data-range-for="' + chartId + '"] [aria-pressed="true"]');
    return pressed ? Number(pressed.dataset.days) : fallback;
  }
  function press(chartId, days) {
    $$('[data-range-for="' + chartId + '"] button').forEach(function (b) {
      b.setAttribute("aria-pressed", b.dataset.days === days ? "true" : "false");
    });
  }
  document.addEventListener("click", function (e) {
    var btn = e.target.closest(".segmented button");
    if (!btn) return;
    var group = btn.parentElement, id = group.dataset.rangeFor;
    press(id, btn.dataset.days);
    (id === "schedule-chart" ? ["schedule-chart", "soc-chart"] : [id]).forEach(function (c) {
      if (charts[c]) charts[c].setDays(Number(btn.dataset.days));
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
            { name: "нагрузка", label: "нагрузка", values: data.load, color: "#8A9290", width: 1.25 },
            { name: "из сети", label: "из сети", values: data.grid, color: "var(--ink)", width: 1.6 }
          ]
        });
        charts["soc-chart"] = new SGChart.TimeChart($("#soc-chart", root), {
          t: data.t, unit: "заряд накопителя, кВт·ч", window: data.window, days: 7,
          series: [{ name: "заряд", label: "заряд", values: data.soc, color: "var(--save)", width: 1.6 }]
        });
      } else if (kind === "monthly") {
        SGChart.BarChart($("#monthly-chart", root), {
          unit: "₽", values: data.map(function (m) { return m.gross_savings; }),
          partial: data.map(function (m) { return m.partial; }),
          labels: data.map(function (m) {
            var p = m.month.split("-");
            return fmt.monthOnly(Date.UTC(+p[0], +p[1] - 1, 1) / 1000) + (m.partial ? "*" : "");
          })
        });
      } else if (kind === "forecast") {
        var band = data.band && data.band.p10 ? { lo: data.band.p10, hi: data.band.p90 } : null;
        charts["forecast-chart"] = new SGChart.TimeChart($("#forecast-chart", root), {
          t: data.t, unit: "кВт·ч", days: 7, band: band,
          series: [
            { name: "факт", label: "факт", values: data.actual, color: "var(--ink)", width: 1.5 },
            { name: "прогноз", label: "прогноз", values: data.forecast, color: "var(--model)", width: 2 }
          ]
        });
      } else if (kind === "profile") {
        SGChart.DayProfile($("#profile-chart", root) || $("#profile-chart"), data);
      } else if (kind === "cashflow") {
        SGChart.Cashflow($("#cashflow-chart", root) || $("#cashflow-chart"), data);
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

  // ── Фоновые расчёты: ток по шине, опрос, остановка ────────────────────────
  function runJob(url, body, target, onDone, onFail) {
    var started = Date.now(), jobId = null, stopped = false;
    var progress = document.createElement("div");
    progress.className = "progress";
    progress.setAttribute("role", "status");
    progress.innerHTML = '<span data-stage>Считаем…</span>' +
      '<button type="button" class="btn small">' + '■ Остановить</button><div class="bar"><i></i></div>';
    target.prepend(progress);
    $("button", progress).addEventListener("click", function () {
      stopped = true;
      if (jobId) fetch("/jobs/" + jobId + "/stop", { method: "POST" });
      $("button", progress).disabled = true;
      $("[data-stage]", progress).textContent = "Останавливаем…";
    });
    function fail(text) {
      progress.outerHTML = errorHtml(text);
      if (onFail) onFail();
    }
    fetch(url, { method: "POST", body: body, headers: { "X-Requested-With": "fetch" } })
      .then(function (r) {
        return r.json().catch(function () { return { detail: "Сервер ответил не так, как ожидалось" }; })
          .then(function (d) { if (!r.ok) throw new Error(d.detail || "Ошибка запроса"); return d; });
      })
      .then(function (d) { jobId = d.job; poll(); })
      .catch(function (err) { fail(err.message); });

    function poll() {
      fetch("/jobs/" + jobId)
        .then(function (r) { if (!r.ok) throw new Error("Расчёт потерян — вероятно, приложение перезапускали. Запустите его снова."); return r.json(); })
        .then(function (s) {
          if (s.status === "running") {
            var secs = Math.round((Date.now() - started) / 1000);
            if (!stopped) $("[data-stage]", progress).textContent = "Считаем: " + (s.stage || "подготовка") +
              " · прошло " + Math.floor(secs / 60) + ":" + ("0" + secs % 60).slice(-2);
            $(".bar i", progress).style.setProperty("--p", Math.max(2, Math.round(s.progress * 100)) + "%");
            setTimeout(poll, 700);
            return;
          }
          progress.remove();
          onDone(s.html || "", s.status);
        })
        .catch(function (err) { fail(err.message); });
    }
  }

  function errorHtml(text) {
    var d = document.createElement("div");
    d.className = "error-box";
    d.setAttribute("role", "alert");
    d.textContent = text;
    return d.outerHTML;
  }

  // «Было / стало»: после пересчёта цифра итога отсчитывается от прежней, а
  // рядом на несколько секунд остаётся прежнее значение — видно, что изменилось.
  function countUp(node, from, to) {
    if (calm || !isFinite(from) || !isFinite(to) || from === to) return;
    var t0 = performance.now(), dur = 700;
    function frame(now) {
      var k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      node.textContent = fmt.rub(from + (to - from) * e);
      if (k < 1) requestAnimationFrame(frame); else node.textContent = fmt.rub(to);
    }
    requestAnimationFrame(frame);
  }

  var econForm = $("#econ-form");
  if (econForm) {
    econForm.addEventListener("submit", function (e) {
      e.preventDefault();
      var target = $("#econ-result"), button = $('button[type="submit"]', econForm);
      var old = $("#readout", target), before = old ? Number(old.dataset.value) : NaN;
      $$(".error-box", target).forEach(function (x) { x.remove(); });
      button.disabled = true;
      runJob(econForm.dataset.run, new FormData(econForm), target, function (html) {
        button.disabled = false;
        target.innerHTML = html;
        target.closest(".split").classList.add("has-result");
        initCharts(target);
        var fresh = $("#readout", target), figure = fresh && $("[data-figure]", fresh);
        if (figure && isFinite(before)) {
          var after = Number(fresh.dataset.value);
          if (Math.abs(after - before) > 0.5) {
            countUp(figure, before, after);
            var chip = document.createElement("span");
            chip.className = "delta";
            chip.textContent = "было " + fmt.rub(before);
            $(".figure-label", fresh).appendChild(chip);
            figure.classList.add("flash");
            setTimeout(function () { chip.remove(); }, 6000);
          }
        }
        var mnemo = $('.mnemo .node[data-node="battery"]');
        if (mnemo) {
          mnemo.classList.remove("lamp-off");
          mnemo.classList.add("lamp-on", "changed");
          var status = $(".node-text span", mnemo);
          if (status && fresh) status.textContent = fmt.rub(Number(fresh.dataset.value)) + (fresh.dataset.annual === "1" ? " в год" : "");
        }
        $$(".mnemo .bus").forEach(function (b) { b.classList.add("live"); });
      }, function () { button.disabled = false; });
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
    }, function () { btn.disabled = false; });
  });

  // ── Финансы: пересчёт NPV и IRR без повторного расчёта накопителя ─────────
  var finTimer = null;
  document.addEventListener("input", function (e) {
    var form = e.target.closest("form[data-finance]");
    if (!form) return;
    clearTimeout(finTimer);
    finTimer = setTimeout(function () {
      var box = $("#finance"), active = document.activeElement && document.activeElement.name;
      fetch(form.dataset.finance, { method: "POST", body: new FormData(form), headers: { "X-Requested-With": "fetch" } })
        .then(function (r) { return r.text().then(function (t) { return { ok: r.ok, text: t }; }); })
        .then(function (res) {
          if (!res.ok) {
            $$(".error-box", box).forEach(function (x) { x.remove(); });
            form.insertAdjacentHTML("afterend", res.text);
            return;
          }
          box.innerHTML = res.text;
          initCharts(box);
          var again = active && $('input[name="' + active + '"]', box);
          if (again) { again.focus(); again.setSelectionRange(again.value.length, again.value.length); }
        });
    }, 450);
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
  var banner = $("#example-banner");
  if (econForm) {
    econForm.appendChild(confirmed);
    if (banner && banner.hidden) confirmed.value = "1";
  }
  // Пометка «пример» держится, пока значение совпадает с примерным: стёр и
  // вернул ту же цифру — это всё ещё пример, а не ставка из счёта.
  $$("input[data-example]").forEach(function (input) {
    var label = input.closest("label");
    function sync() { label.classList.toggle("is-example", confirmed.value !== "1" && input.value === input.dataset.example); }
    input.addEventListener("input", function () { sync(); updateSummaries(); });
    sync();
  });
  var confirmBtn = $("[data-confirm-rates]");
  if (confirmBtn) confirmBtn.addEventListener("click", function () {
    $$("label.is-example").forEach(function (l) { l.classList.remove("is-example"); });
    confirmed.value = "1";
    if (banner) banner.hidden = true;
    updateSummaries();
  });

  // ── Сводки в заголовках секций пульта ─────────────────────────────────────
  function updateSummaries() {
    if (!econForm) return;
    var total = $$('input[name="meters"]', econForm).length;
    var picked = $$('input[name="meters"]:checked', econForm).length;
    var set = function (key, text) { var e = $('[data-summary="' + key + '"]'); if (e) e.textContent = text; };
    set("meters", picked + " из " + total);
    var example = $$("label.is-example:not([hidden])", econForm).length;
    set("tariff", "категория " + (category ? category.value : "") + (example ? " · примерных " + example : " · ваши ставки"));
    var hours = ($('input[name="peak_hours"]', econForm).value || "").split(",").filter(Boolean).map(Number);
    set("hours", hours.length ? hours[0] + "–" + (hours[hours.length - 1] + 1) + " ч" : "не выбраны");
    var share = $('input[name="power_share"]', econForm);
    set("battery", (share ? share.value : "") + " % максимума");
  }
  if (econForm) {
    econForm.addEventListener("input", updateSummaries);
    econForm.addEventListener("change", updateSummaries);
  }

  // ── Шкала суток: плановые часы пиковой нагрузки ───────────────────────────
  document.addEventListener("click", function (e) {
    var btn = e.target.closest("[data-hours]");
    if (!btn) return;
    var p = btn.dataset.hours.split("-").map(Number), hours = [];
    for (var h = p[0]; h < p[1]; h++) hours.push(h);
    var scale = $("[data-dayscale]");
    if (scale && scale.setHours) scale.setHours(hours, true);
  });

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
    function sync(wave) {
      $$("span", scale).forEach(function (c, i) {
        var was = c.classList.contains("on"), now = !!on[+c.dataset.hour];
        c.classList.toggle("on", now);
        if (wave && now && !was && !calm) {
          c.classList.remove("pulse");
          c.style.animationDelay = (i * 18) + "ms";
          void c.offsetWidth;
          c.classList.add("pulse");
        }
      });
      var hours = Object.keys(on).filter(function (k) { return on[k]; }).map(Number).sort(function (a, b) { return a - b; });
      input.value = hours.join(",");
      var gap = hours.length && hours[hours.length - 1] - hours[0] + 1 !== hours.length;
      axis.textContent = !hours.length ? "Часы не выбраны" :
        gap ? "Часы должны идти подряд: окно пика одно на сутки" :
        "Выделено " + hours[0] + ":00–" + (hours[hours.length - 1] + 1) + ":00 (" + hours.length + " ч)";
      axis.style.color = gap || !hours.length ? "var(--err)" : "";
    }
    scale.setHours = function (hours, wave) {
      on = {}; hours.forEach(function (x) { on[x] = true; }); sync(wave); updateSummaries();
    };
    scale.addEventListener("pointerdown", function (e) {
      var c = e.target.closest("span"); if (!c) return;
      dragging = true; mode = !on[+c.dataset.hour]; on[+c.dataset.hour] = mode; sync(); updateSummaries();
    });
    scale.addEventListener("pointerover", function (e) {
      var c = e.target.closest("span"); if (!dragging || !c) return;
      on[+c.dataset.hour] = mode; sync(); updateSummaries();
    });
    document.addEventListener("pointerup", function () { dragging = false; });
    sync();
  });
  updateSummaries();
})();
