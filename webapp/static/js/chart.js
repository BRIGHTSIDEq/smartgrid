// Собственные графики на SVG и canvas — без внешних библиотек (приложение
// работает без интернета). Правила: docs/design/VISUAL_SYSTEM.md, «Графики».
(function () {
  "use strict";
  var SVG = "http://www.w3.org/2000/svg";
  var HOUR = 3600, DAY = 86400;
  var calm = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function el(tag, attrs, parent) {
    var e = document.createElementNS(SVG, tag);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  function token(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function niceStep(span, count) {
    var raw = span / Math.max(count, 1), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var norm = raw / mag;
    return (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  }

  function extent(arrays) {
    var lo = Infinity, hi = -Infinity;
    arrays.forEach(function (a) {
      if (!a) return;
      for (var i = 0; i < a.length; i++) {
        var v = a[i];
        if (v === null || v === undefined || isNaN(v)) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    });
    return [lo, hi];
  }

  function bisect(t, x) {
    var lo = 0, hi = t.length - 1;
    while (hi - lo > 1) { var mid = (lo + hi) >> 1; if (t[mid] < x) lo = mid; else hi = mid; }
    return Math.abs(t[lo] - x) <= Math.abs(t[hi] - x) ? lo : hi;
  }

  // Проявление слева направо: движение отвечает на действие (новый расчёт,
  // другой прибор), а не украшает. При «меньше движения» — сразу целиком.
  function reveal(svg, x, y, w, h) {
    if (calm || !svg.animate) return null;
    var id = "clip" + Math.random().toString(36).slice(2, 8);
    var clip = el("clipPath", { id: id }, el("defs", {}, svg));
    var r = el("rect", { x: x, y: y - 2, width: w, height: h + 4 }, clip);
    r.animate([{ width: 0 }, { width: w }], { duration: 650, easing: "cubic-bezier(.2,.7,.2,1)" });
    return "url(#" + id + ")";
  }

  // Путь линии с разрывами на пропусках. При точках гуще пикселя берутся
  // минимум и максимум на каждый пиксель: форма сохраняется, узлов — вдвое
  // больше ширины графика, а не тысячи.
  function linePath(t, v, i0, i1, sx, sy, width) {
    var d = "", pen = false, n = i1 - i0 + 1;
    if (n > width * 2) {
      var bucket = -1, bmin = null, bmax = null;
      var flush = function (x) {
        if (bmin === null) { pen = false; return; }
        d += (pen ? "L" : "M") + x + "," + sy(bmax) + "L" + x + "," + sy(bmin);
        pen = true;
      };
      for (var i = i0; i <= i1; i++) {
        var x = Math.round(sx(t[i]));
        if (x !== bucket) { if (bucket >= 0) flush(bucket); bucket = x; bmin = bmax = null; }
        var y = v[i];
        if (y === null || y === undefined || isNaN(y)) continue;
        if (bmin === null || y < bmin) bmin = y;
        if (bmax === null || y > bmax) bmax = y;
      }
      flush(bucket);
      return d;
    }
    for (var j = i0; j <= i1; j++) {
      var w = v[j];
      if (w === null || w === undefined || isNaN(w)) { pen = false; continue; }
      d += (pen ? "L" : "M") + sx(t[j]).toFixed(1) + "," + sy(w).toFixed(1);
      pen = true;
    }
    return d;
  }

  function timeTicks(t0, t1, width) {
    var span = t1 - t0, ticks = [];
    if (span <= 2 * DAY) {
      for (var h = Math.ceil(t0 / (6 * HOUR)) * 6 * HOUR; h <= t1; h += 6 * HOUR) {
        var hh = new Date(h * 1000).getUTCHours();
        ticks.push({ t: h, label: hh === 0 ? fmt.date(h) : (hh < 10 ? "0" : "") + hh + ":00", major: hh === 0 });
      }
      return ticks;
    }
    if (span <= 62 * DAY) {
      var every = Math.max(1, Math.ceil((span / DAY) / (width / 64)));
      var start = Math.ceil(t0 / DAY) * DAY, k = 0;
      for (var d = start; d <= t1; d += DAY, k++) {
        ticks.push({ t: d, label: k % every === 0 ? fmt.date(d) : "", major: true });
      }
      return ticks;
    }
    var dt = new Date(t0 * 1000);
    var m = new Date(Date.UTC(dt.getUTCFullYear(), dt.getUTCMonth() + 1, 1));
    var monthsTotal = span / (30 * DAY), step = Math.max(1, Math.ceil(monthsTotal / (width / 70)));
    for (var i = 0; m.getTime() / 1000 <= t1; i++) {
      var s = m.getTime() / 1000;
      if (i % step === 0) ticks.push({ t: s, label: m.getUTCMonth() === 0 ? fmt.month(s) : fmt.monthOnly(s), major: true });
      m = new Date(Date.UTC(m.getUTCFullYear(), m.getUTCMonth() + 1, 1));
    }
    return ticks;
  }

  function yAxis(svg, m, pw, sy, ymin, ymax, step, unit) {
    var grid = el("g", { "class": "grid axis" }, svg);
    for (var y = ymin; y <= ymax + step / 2; y += step) {
      el("line", { x1: m.l, x2: m.l + pw, y1: sy(y), y2: sy(y), "shape-rendering": "crispEdges" }, grid);
      el("text", { x: m.l - 8, y: sy(y) + 4, "text-anchor": "end" }, grid).textContent = fmt.axis(y, step);
    }
    el("text", { "class": "unit", x: 8, y: 14 }, svg).textContent = unit || "";
  }

  // Подсказка-бирка: тёмная табличка у перекрестья, отражается у края.
  function makeTip(host) {
    var tip = document.createElement("div");
    tip.className = "tip";
    tip.hidden = true;
    host.appendChild(tip);
    return {
      show: function (html, cx) {
        tip.innerHTML = html;
        tip.hidden = false;
        var left = cx + 14, side = "right";
        if (left + tip.offsetWidth > host.clientWidth - 4) { left = cx - tip.offsetWidth - 14; side = "left"; }
        tip.className = "tip " + side;
        tip.style.left = left + "px";
      },
      hide: function () { tip.hidden = true; }
    };
  }

  /**
   * Временной ряд.
   * opts: {t, series:[{name, values, color, width, dash, label}], band:{lo,hi},
   *        unit, window:[from,to] часы пиковой нагрузки (только будни),
   *        problems:[t], forecastStart:t, days: окно в сутках (0 — всё)}
   */
  function TimeChart(host, opts) {
    this.host = host;
    this.opts = opts;
    this.days = opts.days || 0;
    this.fresh = true;
    var self = this;
    if (window.ResizeObserver) {
      new ResizeObserver(function () { self.render(); }).observe(host);
    }
    this.render();
  }

  TimeChart.prototype.setDays = function (days) { this.days = days; this.anchor = null; this.fresh = true; this.render(); };
  // Окно вокруг момента: например, чтобы открыть прибор на участке с проблемой.
  TimeChart.prototype.focus = function (t, days) { this.anchor = t; this.days = days; this.fresh = true; this.render(); };

  TimeChart.prototype.render = function () {
    var o = this.opts, host = this.host, t = o.t;
    host.innerHTML = "";
    var W = host.clientWidth, H = host.clientHeight;
    if (!W || !H || !t || !t.length) return;
    var labels = o.series.filter(function (s) { return s.label; }).length;
    var m = { l: 56, r: labels ? 104 : 16, t: 24, b: o.problems ? 34 : 26 };
    var pw = W - m.l - m.r, ph = H - m.t - m.b;

    var tEnd = t[t.length - 1] + HOUR, tStart = this.days ? Math.max(t[0], tEnd - this.days * DAY) : t[0];
    if (this.anchor && this.days) {
      tStart = Math.max(t[0], this.anchor - 2 * DAY);
      tEnd = Math.min(t[t.length - 1] + HOUR, tStart + this.days * DAY);
    }
    var i0 = bisect(t, tStart), i1 = bisect(t, tEnd - HOUR);
    if (t[i0] < tStart && i0 < i1) i0++;
    var visible = o.series.map(function (s) { return s.values.slice(i0, i1 + 1); });
    if (o.band) { visible.push(o.band.lo.slice(i0, i1 + 1)); visible.push(o.band.hi.slice(i0, i1 + 1)); }
    var ex = extent(visible), ymin = Math.min(0, ex[0]), ymax = ex[1] > ymin ? ex[1] : ymin + 1;
    var step = niceStep(ymax - ymin, Math.max(3, Math.floor(ph / 60)));
    ymax = Math.ceil(ymax / step) * step;
    ymin = Math.floor(ymin / step) * step;

    var sx = function (x) { return m.l + (x - tStart) / (tEnd - tStart) * pw; };
    var sy = function (y) { return m.t + ph - (y - ymin) / (ymax - ymin) * ph; };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": o.aria || "" }, host);

    // Плановые часы пиковой нагрузки — только будни: в выходные мощность не оплачивается.
    if (o.window) {
      var g = el("g", { "class": "peaks" }, svg);
      for (var d = Math.floor(tStart / DAY) * DAY; d < tEnd; d += DAY) {
        var wd = new Date(d * 1000).getUTCDay();
        if (wd === 0 || wd === 6) continue;
        var a = Math.max(d + o.window[0] * HOUR, tStart), b = Math.min(d + o.window[1] * HOUR, tEnd);
        if (b > a) el("rect", { "class": "peak", x: sx(a), y: m.t, width: Math.max(sx(b) - sx(a), 1), height: ph }, g);
      }
    }

    yAxis(svg, m, pw, sy, ymin, ymax, step, o.unit);
    var xa = el("g", { "class": "axis" }, svg);
    timeTicks(tStart, tEnd, pw).forEach(function (tk) {
      var x = sx(tk.t);
      if (x < m.l - 1 || x > m.l + pw + 1) return;
      el("line", { x1: x, x2: x, y1: m.t + ph, y2: m.t + ph + 4, stroke: "var(--rule-strong)" }, xa);
      if (tk.label) el("text", { x: x, y: m.t + ph + 17, "text-anchor": "middle" }, xa).textContent = tk.label;
    });

    var clip = this.fresh ? reveal(svg, m.l, m.t, pw, ph) : null;
    this.fresh = false;
    var plot = el("g", { "clip-path": clip }, svg);

    if (o.band) {
      var top = "", bottom = "";
      for (var i = i0; i <= i1; i++) {
        if (o.band.lo[i] === null || o.band.hi[i] === null) continue;
        top += (top ? "L" : "M") + sx(t[i]).toFixed(1) + "," + sy(o.band.hi[i]).toFixed(1);
        bottom = "L" + sx(t[i]).toFixed(1) + "," + sy(o.band.lo[i]).toFixed(1) + bottom;
      }
      if (top) el("path", { d: top + bottom + "Z", fill: "var(--model-band)" }, plot);
    }

    var ends = [];
    o.series.forEach(function (s) {
      el("path", { d: linePath(t, s.values, i0, i1, sx, sy, pw), fill: "none", stroke: s.color,
                   "stroke-width": s.width || 1.5, "stroke-dasharray": s.dash || null,
                   "stroke-linejoin": "round" }, plot);
      if (!s.label) return;
      for (var k = i1; k >= i0; k--) {
        if (s.values[k] !== null && !isNaN(s.values[k])) { ends.push({ y: sy(s.values[k]), s: s }); break; }
      }
    });
    // Подписи на концах линий вместо легенды; при столкновении раздвигаются.
    ends.sort(function (a, b) { return a.y - b.y; });
    for (var e = 1; e < ends.length; e++) if (ends[e].y - ends[e - 1].y < 15) ends[e].y = ends[e - 1].y + 15;
    ends.forEach(function (end) {
      el("line", { x1: m.l + pw + 3, x2: m.l + pw + 13, y1: end.y, y2: end.y, stroke: end.s.color,
                   "stroke-width": 2, "stroke-dasharray": end.s.dash || null }, svg);
      el("text", { "class": "line-label", x: m.l + pw + 17, y: end.y + 4, fill: "var(--ink)" }, svg).textContent = end.s.label;
    });

    if (o.forecastStart && o.forecastStart > tStart && o.forecastStart < tEnd) {
      var fx = sx(o.forecastStart);
      el("line", { x1: fx, x2: fx, y1: m.t, y2: m.t + ph, stroke: "var(--ink)" }, svg);
      el("text", { x: fx + 4, y: m.t + 12, fill: "var(--ink-2)" }, svg).textContent = "начало прогноза";
    }

    if (o.problems && o.problems.length) {
      // Проблемные часы — полупрозрачная заливка на всю высоту и отметка под осью.
      var pg = el("g", {}, svg), hourW = Math.max(pw / ((tEnd - tStart) / HOUR), 2);
      o.problems.forEach(function (p) {
        if (p < tStart || p >= tEnd) return;
        el("rect", { "class": "problem-zone", x: sx(p), y: m.t, width: hourW, height: ph }, pg);
        el("rect", { "class": "problem", x: sx(p), y: m.t + ph + 24, width: hourW, height: 5 }, pg);
      });
    }

    this.attachTip(svg, { t: t, i0: i0, i1: i1, sx: sx, m: m, pw: pw, ph: ph, tStart: tStart, tEnd: tEnd });
  };

  TimeChart.prototype.attachTip = function (svg, g) {
    var o = this.opts, tip = makeTip(this.host);
    var cross = el("line", { y1: g.m.t, y2: g.m.t + g.ph, stroke: "var(--ink-2)", "stroke-dasharray": "2 3", visibility: "hidden" }, svg);
    svg.addEventListener("mousemove", function (ev) {
      var r = svg.getBoundingClientRect(), x = ev.clientX - r.left;
      if (x < g.m.l || x > g.m.l + g.pw) { cross.setAttribute("visibility", "hidden"); tip.hide(); return; }
      var tx = g.tStart + (x - g.m.l) / g.pw * (g.tEnd - g.tStart);
      var i = Math.min(Math.max(bisect(g.t, tx), g.i0), g.i1), cx = g.sx(g.t[i]);
      cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
      var hour = new Date(g.t[i] * 1000), wd = hour.getUTCDay(), h = hour.getUTCHours();
      var inPeak = o.window && wd !== 0 && wd !== 6 && h >= o.window[0] && h < o.window[1];
      var rows = "<b>" + fmt.hourSpan(g.t[i]) + (inPeak ? " · пик" : "") + "</b>";
      o.series.forEach(function (s) {
        rows += '<div class="row"><span><i class="sw" style="background:' + s.color + '"></i>' + (s.tipName || s.name) +
                "</span><span>" + fmt.num(s.values[i], s.digits === undefined ? 1 : s.digits) + "</span></div>";
      });
      if (o.band && o.band.lo[i] !== null) {
        rows += '<div class="row"><span>коридор</span><span>' + fmt.num(o.band.lo[i], 1) + "–" +
                fmt.num(o.band.hi[i], 1) + "</span></div>";
      }
      tip.show(rows, cx);
    });
    svg.addEventListener("mouseleave", function () { cross.setAttribute("visibility", "hidden"); tip.hide(); });
  };

  /** Столбики от нуля: {labels, values, unit, partial}. Отрицательные — серые, вниз. */
  function BarChart(host, opts) {
    host.innerHTML = "";
    var W = host.clientWidth, H = host.clientHeight, m = { l: 56, r: 16, t: 22, b: 24 };
    var pw = W - m.l - m.r, ph = H - m.t - m.b;
    var ex = extent([opts.values]), lo = Math.min(0, ex[0]), hi = Math.max(0, ex[1]);
    var step = niceStep(hi - lo || 1, 3);
    hi = Math.ceil(hi / step) * step; lo = Math.floor(lo / step) * step;
    var sy = function (v) { return m.t + ph - (v - lo) / (hi - lo || 1) * ph; };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H }, host);
    yAxis(svg, m, pw, sy, lo, hi, step, opts.unit);
    var n = opts.values.length, bw = pw / n, tip = makeTip(host);
    opts.values.forEach(function (v, i) {
      var x = m.l + i * bw + bw * 0.18, y0 = sy(0), y1 = sy(v);
      var faint = opts.partial && opts.partial[i];
      var bar = el("rect", { x: x, y: Math.min(y0, y1), width: bw * 0.64, height: Math.max(Math.abs(y1 - y0), 1),
                             fill: v < 0 ? "var(--rule-strong)" : faint ? "#A9CDB4" : "var(--save)" }, svg);
      if (!calm && bar.animate) {
        bar.style.transformOrigin = "0 " + y0 + "px";
        bar.animate([{ transform: "scaleY(0)" }, { transform: "scaleY(1)" }],
                    { duration: 420, delay: i * 35, easing: "cubic-bezier(.2,.7,.2,1)", fill: "backwards" });
      }
      bar.addEventListener("mouseenter", function () {
        tip.show("<b>" + opts.labels[i] + "</b><div class='row'><span>экономия</span><span>" + fmt.rub(v) + "</span></div>", x + bw * 0.64);
      });
      bar.addEventListener("mouseleave", tip.hide);
      el("text", { x: x + bw * 0.32, y: H - 8, "text-anchor": "middle", fill: "var(--ink-3)", "font-size": 11 }, svg).textContent = opts.labels[i];
    });
  }

  /**
   * Разброс. Крупный — шкала прибора: деления, закрашенный разброс и стрелка
   * на оценке; маленький — отрезок с засечками для таблиц.
   */
  function Spread(host, lo, hi, mid, domain, fromZero) {
    host.innerHTML = "";
    var big = host.dataset.big === "1";
    var W = host.clientWidth || 140, H = big ? 46 : 12;
    if (fromZero) domain = [Math.min(0, lo), Math.max(hi, 0) * 1.1 || 1];
    // Своя шкала — вокруг самого разброса; ноль попадает на шкалу, только
    // если разброс уходит в минус. Иначе узкий разброс крупной суммы
    // сжимался бы в точку у правого края.
    var width = hi - lo || Math.abs(mid) * 0.05 || 1;
    var d0 = domain ? domain[0] : (lo < 0 ? Math.min(lo, 0) - width * 0.3 : lo - width * 0.6);
    var d1 = domain ? domain[1] : hi + width * 0.6;
    if (d1 <= d0) d1 = d0 + 1;
    var sx = function (v) { return 6 + (v - d0) / (d1 - d0) * (W - 12); };
    var svg = el("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H }, host);
    if (big) {
      var base = 22, step = niceStep(d1 - d0, Math.max(4, Math.floor(W / 70)));
      el("line", { x1: sx(d0), x2: sx(d1), y1: base, y2: base, stroke: "var(--ink)", "stroke-width": 1.5 }, svg);
      for (var v = Math.ceil(d0 / (step / 5)) * (step / 5); v <= d1 + 1e-9; v += step / 5) {
        var major = Math.abs(v / step - Math.round(v / step)) < 1e-6;
        el("line", { x1: sx(v), x2: sx(v), y1: base, y2: base + (major ? 8 : 4), stroke: "var(--ink-2)" }, svg);
        if (major) el("text", { x: sx(v), y: H - 1, "text-anchor": "middle" }, svg).textContent = fmt.axis(v, step);
      }
      var band = el("rect", { x: sx(lo), y: base - 9, width: Math.max(sx(hi) - sx(lo), 3), height: 8,
                              fill: "var(--save-tint)", stroke: "var(--save)" }, svg);
      var needle = el("path", { d: "M" + sx(mid) + "," + (base + 1) + "l-6,-13h12z", fill: "var(--ink)" }, svg);
      if (!calm && needle.animate) {
        needle.animate([{ transform: "translateX(" + (sx(d0) - sx(mid)) + "px)" }, { transform: "translateX(0)" }],
                       { duration: 900, easing: "cubic-bezier(.3,1.35,.5,1)" });
        band.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 500, delay: 300, fill: "backwards" });
      }
    } else {
      if (d0 < 0 && d1 > 0) el("line", { x1: sx(0), x2: sx(0), y1: 0, y2: H, stroke: "var(--ink-3)" }, svg);
      el("line", { x1: sx(lo), x2: sx(hi), y1: H / 2, y2: H / 2, stroke: "var(--ink-2)", "stroke-width": 2 }, svg);
      el("line", { x1: sx(lo), x2: sx(lo), y1: 1, y2: H - 1, stroke: "var(--ink-2)" }, svg);
      el("line", { x1: sx(hi), x2: sx(hi), y1: 1, y2: H - 1, stroke: "var(--ink-2)" }, svg);
      el("rect", { x: sx(mid) - 3, y: H / 2 - 3, width: 6, height: 6, fill: "var(--ink)" }, svg);
    }
    host.title = fmt.rub(lo) + " … " + fmt.rub(hi);
  }

  /**
   * Ковёр нагрузки: столбец — сутки, строка — час. Инженер видит год целиком:
   * рабочие недели, праздники, пропуски, сдвиги режима. Проблемные часы — красным.
   */
  function Carpet(wrap, data, onPick) {
    var canvas = wrap.querySelector("canvas"), axis = wrap.querySelector(".carpet-axis");
    var t = data.t, y = data.y;
    if (!t.length) return;
    var day0 = Math.floor(t[0] / DAY), nDays = Math.floor(t[t.length - 1] / DAY) - day0 + 1;
    var grid = new Float32Array(nDays * 24).fill(NaN), bad = new Uint8Array(nDays * 24);
    var vals = [];
    for (var i = 0; i < t.length; i++) {
      var d = Math.floor(t[i] / DAY) - day0, h = new Date(t[i] * 1000).getUTCHours();
      if (y[i] !== null) { grid[d * 24 + h] = y[i]; vals.push(y[i]); }
    }
    (data.problems || []).forEach(function (p) {
      var d = Math.floor(p / DAY) - day0, h = new Date(p * 1000).getUTCHours();
      if (d >= 0 && d < nDays) bad[d * 24 + h] = 1;
    });
    vals.sort(function (a, b) { return a - b; });
    var q = function (p) { return vals.length ? vals[Math.min(vals.length - 1, Math.floor(p * vals.length))] : 0; };
    var lo = q(0.02), hi = q(0.98) || 1;
    // Одна краска по яркости: от светлой бумаги к чернильно-зелёному.
    var stops = [[244, 245, 241], [157, 179, 164], [62, 94, 74], [17, 24, 18]];
    function colour(v) {
      var x = Math.max(0, Math.min(1, (v - lo) / (hi - lo || 1))) * (stops.length - 1);
      var k = Math.min(stops.length - 2, Math.floor(x)), f = x - k;
      return stops[k].map(function (c, j) { return Math.round(c + (stops[k + 1][j] - c) * f); });
    }
    function draw() {
      var cw = canvas.clientWidth, ch = canvas.clientHeight;
      if (!cw) return;
      canvas.width = nDays; canvas.height = 24;
      var ctx = canvas.getContext("2d"), img = ctx.createImageData(nDays, 24);
      for (var d = 0; d < nDays; d++) {
        for (var h = 0; h < 24; h++) {
          var v = grid[d * 24 + h], p = (h * nDays + d) * 4, c;
          if (bad[d * 24 + h]) c = [179, 38, 30];
          else if (isNaN(v)) c = [255, 255, 255];
          else c = colour(v);
          img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      if (!calm && canvas.animate && !canvas.dataset.shown) {
        canvas.animate([{ clipPath: "inset(0 100% 0 0)" }, { clipPath: "inset(0 0 0 0)" }], { duration: 700, easing: "ease-out" });
      }
      canvas.dataset.shown = "1";
    }
    draw();
    if (axis) {
      axis.innerHTML = "";
      var marks = 6;
      for (var k = 0; k <= marks; k++) {
        var s = document.createElement("span");
        s.textContent = fmt.date((day0 + Math.round(k * (nDays - 1) / marks)) * DAY);
        axis.appendChild(s);
      }
    }
    canvas.onclick = function (ev) {
      var r = canvas.getBoundingClientRect();
      var d = Math.floor((ev.clientX - r.left) / r.width * nDays);
      if (onPick) onPick((day0 + d) * DAY);
    };
    canvas.onmousemove = function (ev) {
      var r = canvas.getBoundingClientRect();
      var d = Math.floor((ev.clientX - r.left) / r.width * nDays), h = Math.floor((ev.clientY - r.top) / r.height * 24);
      var v = grid[d * 24 + h];
      canvas.title = fmt.hourSpan((day0 + d) * DAY + h * HOUR) + ": " + (isNaN(v) ? "нет данных" : fmt.num(v, 1) + " кВт·ч") +
                     (bad[d * 24 + h] ? " — проблемный час" : "");
    };
  }

  /**
   * Средние рабочие сутки: нагрузка и взятое из сети по часам. Срезанное в
   * часы пика закрашено зелёным — это и есть то, за что платят меньше.
   */
  function DayProfile(host, data) {
    host.innerHTML = "";
    var W = host.clientWidth, H = host.clientHeight;
    if (!W || !H) return;
    var m = { l: 48, r: 8, t: 26, b: 22 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
    var ex = extent([data.load, data.grid]), ymax = ex[1] || 1, step = niceStep(ymax, 3);
    ymax = Math.ceil(ymax / step) * step;
    var sx = function (h) { return m.l + h / 24 * pw; };
    var sy = function (v) { return m.t + ph - v / ymax * ph; };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Средние рабочие сутки" }, host);
    if (data.window) el("rect", { "class": "peak", x: sx(data.window[0]), y: m.t, width: sx(data.window[1]) - sx(data.window[0]), height: ph }, svg);
    yAxis(svg, m, pw, sy, 0, ymax, step, "кВт·ч");
    var xa = el("g", { "class": "axis" }, svg);
    for (var h = 0; h <= 24; h += 6) el("text", { x: sx(h), y: H - 6, "text-anchor": "middle" }, xa).textContent = h + " ч";
    // Ступеньки: час учёта — это целый час, а не точка.
    function steps(v) {
      var d = "";
      for (var i = 0; i < 24; i++) d += (i ? "L" : "M") + sx(i) + "," + sy(v[i]) + "L" + sx(i + 1) + "," + sy(v[i]);
      return d;
    }
    var clip = reveal(svg, m.l, m.t, pw, ph), plot = el("g", { "clip-path": clip }, svg);
    for (var k = 0; k < 24; k++) {
      var cut = data.load[k] - data.grid[k];
      if (Math.abs(cut) < ymax * 0.004) continue;
      el("rect", { x: sx(k), width: sx(k + 1) - sx(k), y: sy(Math.max(data.load[k], data.grid[k])),
                   height: Math.abs(sy(data.load[k]) - sy(data.grid[k])),
                   fill: cut > 0 ? "var(--save-tint)" : "rgba(227,162,26,.18)",
                   stroke: cut > 0 ? "var(--save)" : "none", "stroke-width": .5 }, plot);
    }
    el("path", { d: steps(data.load), fill: "none", stroke: "#8A9290", "stroke-width": 1.5 }, plot);
    el("path", { d: steps(data.grid), fill: "none", stroke: "var(--ink)", "stroke-width": 2 }, plot);
    var tip = makeTip(host);
    svg.addEventListener("mousemove", function (ev) {
      var r = svg.getBoundingClientRect(), i = Math.floor((ev.clientX - r.left - m.l) / pw * 24);
      if (i < 0 || i > 23) { tip.hide(); return; }
      tip.show("<b>" + i + "–" + (i + 1) + " ч, будни</b>" +
               "<div class='row'><span>нагрузка</span><span>" + fmt.num(data.load[i], 1) + "</span></div>" +
               "<div class='row'><span>из сети</span><span>" + fmt.num(data.grid[i], 1) + "</span></div>", sx(i + 1));
    });
    svg.addEventListener("mouseleave", tip.hide);
  }

  /** Накопленный дисконтированный поток по годам с разбросом и точкой окупаемости. */
  function Cashflow(host, data) {
    host.innerHTML = "";
    var W = host.clientWidth, H = host.clientHeight;
    if (!W || !H) return;
    var n = data.mean.length - 1, m = { l: 64, r: 16, t: 16, b: 24 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
    var ex = extent([data.mean, data.lo, data.hi]), lo = Math.min(0, ex[0]), hi = Math.max(0, ex[1]);
    var step = niceStep(hi - lo || 1, 4);
    lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
    var sx = function (y) { return m.l + y / n * pw; }, sy = function (v) { return m.t + ph - (v - lo) / (hi - lo) * ph; };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H }, host);
    yAxis(svg, m, pw, sy, lo, hi, step, "₽");
    el("line", { x1: m.l, x2: m.l + pw, y1: sy(0), y2: sy(0), stroke: "var(--ink)", "stroke-width": 1 }, svg);
    var xa = el("g", { "class": "axis" }, svg), every = Math.max(1, Math.ceil(n / (pw / 40)));
    for (var y = 0; y <= n; y += every) el("text", { x: sx(y), y: H - 6, "text-anchor": "middle" }, xa).textContent = y ? y + " г" : "сейчас";
    var clip = reveal(svg, m.l, m.t, pw, ph), plot = el("g", { "clip-path": clip }, svg);
    var top = "", bottom = "";
    for (var i = 0; i <= n; i++) {
      top += (i ? "L" : "M") + sx(i) + "," + sy(data.hi[i]);
      bottom = "L" + sx(i) + "," + sy(data.lo[i]) + bottom;
    }
    el("path", { d: top + bottom + "Z", fill: "var(--save-tint)" }, plot);
    var d = "";
    for (var j = 0; j <= n; j++) d += (j ? "L" : "M") + sx(j) + "," + sy(data.mean[j]);
    el("path", { d: d, fill: "none", stroke: "var(--ink)", "stroke-width": 2 }, plot);
    for (var k = 0; k <= n; k++) el("rect", { x: sx(k) - 2.5, y: sy(data.mean[k]) - 2.5, width: 5, height: 5,
                                            fill: data.mean[k] < 0 ? "var(--plate)" : "var(--ink)", stroke: "var(--ink)" }, plot);
    if (data.payback) {
      var px = sx(data.payback);
      el("line", { x1: px, x2: px, y1: m.t, y2: m.t + ph, stroke: "var(--save)", "stroke-dasharray": "3 3" }, svg);
      el("text", { x: px + 5, y: m.t + 10, fill: "var(--save)", "font-size": 12 }, svg).textContent = "окупился";
    }
  }

  window.SGChart = { TimeChart: TimeChart, BarChart: BarChart, Spread: Spread, Carpet: Carpet,
                     DayProfile: DayProfile, Cashflow: Cashflow, calm: calm, token: token };
})();
