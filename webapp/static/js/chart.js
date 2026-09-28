// Собственные графики на SVG — без внешних библиотек (приложение работает без
// интернета). Правила оформления: docs/design/VISUAL_SYSTEM.md, раздел «Графики».
(function () {
  "use strict";
  var SVG = "http://www.w3.org/2000/svg";
  var HOUR = 3600, DAY = 86400;

  function el(tag, attrs, parent) {
    var e = document.createElementNS(SVG, tag);
    for (var k in attrs) if (attrs[k] !== undefined && attrs[k] !== null) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
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
    var self = this;
    if (window.ResizeObserver) {
      new ResizeObserver(function () { self.render(); }).observe(host);
    }
    this.render();
  }

  TimeChart.prototype.setDays = function (days) { this.days = days; this.render(); };
  // Окно вокруг момента: например, чтобы открыть прибор на участке с проблемой.
  TimeChart.prototype.focus = function (t, days) { this.anchor = t; this.days = days; this.render(); };

  TimeChart.prototype.render = function () {
    var o = this.opts, host = this.host, t = o.t;
    host.innerHTML = "";
    var W = host.clientWidth, H = host.clientHeight;
    if (!W || !H || !t || !t.length) return;
    var labels = o.series.filter(function (s) { return s.label; }).length;
    var m = { l: 56, r: labels ? 118 : 16, t: 24, b: o.problems ? 34 : 26 };
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

    var grid = el("g", { "class": "grid axis" }, svg);
    for (var y = ymin; y <= ymax + step / 2; y += step) {
      el("line", { x1: m.l, x2: m.l + pw, y1: sy(y), y2: sy(y), "shape-rendering": "crispEdges" }, grid);
      el("text", { x: m.l - 8, y: sy(y) + 4, "text-anchor": "end" }, grid).textContent = fmt.axis(y, step);
    }
    el("text", { "class": "unit", x: 8, y: 14 }, svg).textContent = o.unit || "";

    var xa = el("g", { "class": "axis" }, svg);
    timeTicks(tStart, tEnd, pw).forEach(function (tk) {
      var x = sx(tk.t);
      if (x < m.l - 1 || x > m.l + pw + 1) return;
      el("line", { x1: x, x2: x, y1: m.t + ph, y2: m.t + ph + 4, stroke: "var(--line-strong)" }, xa);
      if (tk.label) el("text", { x: x, y: m.t + ph + 17, "text-anchor": "middle" }, xa).textContent = tk.label;
    });

    if (o.band) {
      var top = "", bottom = "";
      for (var i = i0; i <= i1; i++) {
        if (o.band.lo[i] === null || o.band.hi[i] === null) continue;
        top += (top ? "L" : "M") + sx(t[i]).toFixed(1) + "," + sy(o.band.hi[i]).toFixed(1);
        bottom = "L" + sx(t[i]).toFixed(1) + "," + sy(o.band.lo[i]).toFixed(1) + bottom;
      }
      if (top) el("path", { d: top + bottom + "Z", fill: "var(--accent-band)" }, svg);
    }

    var ends = [];
    o.series.forEach(function (s) {
      el("path", { d: linePath(t, s.values, i0, i1, sx, sy, pw), fill: "none", stroke: s.color,
                   "stroke-width": s.width || 1.5, "stroke-dasharray": s.dash || null,
                   "stroke-linejoin": "round" }, svg);
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
    var o = this.opts, host = this.host;
    var cross = el("line", { y1: g.m.t, y2: g.m.t + g.ph, stroke: "var(--line-strong)", visibility: "hidden" }, svg);
    var tip = document.createElement("div");
    tip.className = "tip";
    tip.hidden = true;
    host.appendChild(tip);
    svg.addEventListener("mousemove", function (ev) {
      var r = svg.getBoundingClientRect(), x = ev.clientX - r.left;
      if (x < g.m.l || x > g.m.l + g.pw) { cross.setAttribute("visibility", "hidden"); tip.hidden = true; return; }
      var tx = g.tStart + (x - g.m.l) / g.pw * (g.tEnd - g.tStart);
      var i = Math.min(Math.max(bisect(g.t, tx), g.i0), g.i1), cx = g.sx(g.t[i]);
      cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.setAttribute("visibility", "visible");
      var rows = "<b>" + fmt.hourSpan(g.t[i]) + "</b>";
      o.series.forEach(function (s) {
        rows += '<div class="row"><span>' + (s.tipName || s.name) + "</span><span>" +
                fmt.num(s.values[i], s.digits === undefined ? 1 : s.digits) + "</span></div>";
      });
      if (o.band && o.band.lo[i] !== null) {
        rows += '<div class="row"><span>коридор</span><span>' + fmt.num(o.band.lo[i], 1) + "–" +
                fmt.num(o.band.hi[i], 1) + "</span></div>";
      }
      tip.innerHTML = rows;
      tip.hidden = false;
      var left = cx + 12;
      if (left + tip.offsetWidth > host.clientWidth) left = cx - tip.offsetWidth - 12;
      tip.style.left = left + "px";
    });
    svg.addEventListener("mouseleave", function () { cross.setAttribute("visibility", "hidden"); tip.hidden = true; });
  };

  /** Столбики от нуля: {labels, values, unit}. Отрицательные — серые, вниз. */
  function BarChart(host, opts) {
    host.innerHTML = "";
    var W = host.clientWidth, H = host.clientHeight, m = { l: 56, r: 16, t: 22, b: 24 };
    var pw = W - m.l - m.r, ph = H - m.t - m.b;
    var ex = extent([opts.values]), lo = Math.min(0, ex[0]), hi = Math.max(0, ex[1]);
    var step = niceStep(hi - lo || 1, 3);
    hi = Math.ceil(hi / step) * step; lo = Math.floor(lo / step) * step;
    var sy = function (v) { return m.t + ph - (v - lo) / (hi - lo || 1) * ph; };
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H }, host);
    var grid = el("g", { "class": "grid axis" }, svg);
    for (var y = lo; y <= hi + step / 2; y += step) {
      el("line", { x1: m.l, x2: m.l + pw, y1: sy(y), y2: sy(y) }, grid);
      el("text", { x: m.l - 8, y: sy(y) + 4, "text-anchor": "end" }, grid).textContent = fmt.axis(y, step);
    }
    el("text", { "class": "unit", x: 8, y: 13 }, svg).textContent = opts.unit || "";
    var n = opts.values.length, bw = pw / n;
    opts.values.forEach(function (v, i) {
      var x = m.l + i * bw + bw * 0.2, y0 = sy(0), y1 = sy(v);
      var faint = opts.partial && opts.partial[i];
      el("rect", { x: x, y: Math.min(y0, y1), width: bw * 0.6, height: Math.max(Math.abs(y1 - y0), 1),
                   rx: 2, fill: v < 0 ? "var(--line-strong)" : faint ? "#A9C6A0" : "var(--ok)" }, svg);
      el("text", { x: x + bw * 0.3, y: H - 8, "text-anchor": "middle", fill: "var(--ink-3)" }, svg).textContent = opts.labels[i];
    });
  }

  /** Полоса разброса: линия от края до края, точка — оценка, засечка — ноль. */
  function Spread(host, lo, hi, mid, domain, fromZero) {
    host.innerHTML = "";
    var big = host.clientHeight >= 24;
    var W = host.clientWidth || 140, H = big ? 30 : 10, bar = big ? 8 : H / 2;
    if (fromZero) domain = [Math.min(0, lo), Math.max(hi, 0) * 1.08 || 1];
    // Своя шкала — вокруг самого разброса; ноль попадает на шкалу, только
    // если разброс уходит в минус. Иначе узкий разброс крупной суммы
    // сжимался бы в точку у правого края.
    var width = hi - lo || Math.abs(mid) * 0.05 || 1;
    var d0 = domain ? domain[0] : (lo < 0 ? Math.min(lo, 0) - width * 0.3 : lo - width * 0.6);
    var d1 = domain ? domain[1] : hi + width * 0.6;
    if (d1 <= d0) d1 = d0 + 1;
    var sx = function (v) { return 4 + (v - d0) / (d1 - d0) * (W - 8); };
    var svg = el("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H }, host);
    if (big) {
      // Крупная шкала: серая дорожка от нуля, закрашенный разброс, подписи.
      el("rect", { x: sx(d0), y: bar - 4, width: sx(d1) - sx(d0), height: 8, rx: 4, fill: "var(--surface-2)" }, svg);
      el("rect", { x: sx(lo), y: bar - 4, width: Math.max(sx(hi) - sx(lo), 3), height: 8, rx: 4, fill: "var(--ok)" }, svg);
      el("circle", { cx: sx(mid), cy: bar, r: 5, fill: "var(--surface)", stroke: "var(--ink)", "stroke-width": 2 }, svg);
      el("text", { x: sx(d0), y: H - 1, "font-size": 11, fill: "var(--ink-3)" }, svg).textContent = "0";
      el("text", { x: sx(hi), y: H - 1, "font-size": 11, fill: "var(--ink-3)", "text-anchor": "end" }, svg).textContent = fmt.rub(hi);
    } else {
      if (d0 < 0 && d1 > 0) el("line", { x1: sx(0), x2: sx(0), y1: 0, y2: H, stroke: "var(--ink-3)" }, svg);
      el("line", { x1: sx(lo), x2: sx(hi), y1: H / 2, y2: H / 2, stroke: "var(--ink-2)", "stroke-width": 2 }, svg);
      el("line", { x1: sx(lo), x2: sx(lo), y1: 1, y2: H - 1, stroke: "var(--ink-2)" }, svg);
      el("line", { x1: sx(hi), x2: sx(hi), y1: 1, y2: H - 1, stroke: "var(--ink-2)" }, svg);
      el("circle", { cx: sx(mid), cy: H / 2, r: 3, fill: "var(--ink)" }, svg);
    }
    host.title = fmt.rub(lo) + " … " + fmt.rub(hi);
  }

  window.SGChart = { TimeChart: TimeChart, BarChart: BarChart, Spread: Spread };
})();
