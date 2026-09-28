// Форматирование чисел и дат — те же правила, что в webapp/formatting.py:
// десятичная запятая, разряды узким неразрывным пробелом, минус U+2212.
(function () {
  "use strict";
  var NNBSP = " ", NBSP = " ", MINUS = "−";
  var MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"];
  var DAYS = ["вс", "пн", "вт", "ср", "чт", "пт", "сб"];

  function num(v, digits, groupFrom) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    if (!isFinite(v)) return "∞";
    digits = digits || 0;
    groupFrom = groupFrom === undefined ? 1000 : groupFrom;
    var a = Math.abs(v), s = a.toFixed(digits), parts = s.split(".");
    if (a >= groupFrom) parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, NNBSP);
    s = parts.join(",");
    var zero = Number(a.toFixed(digits)) === 0;
    return (v < 0 && !zero ? MINUS : "") + s;
  }

  function sig3(v) {
    if (v === 0) return "0";
    var d = Math.max(0, 2 - Math.floor(Math.log10(Math.abs(v))));
    return num(v, d);
  }

  function rub(v) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    var a = Math.abs(v);
    if (a >= 1e6) return sig3(v / 1e6) + NBSP + "млн" + NBSP + "₽";
    if (a >= 1e4) return sig3(v / 1e3) + NBSP + "тыс." + NBSP + "₽";
    return num(v) + NBSP + "₽";
  }

  function axis(v, step) {
    // Подписи осей: точность по шагу делений, крупные числа сокращаются.
    var a = Math.abs(v);
    if (a >= 1e6) return num(v / 1e6, step >= 1e6 ? 0 : 1) + NBSP + "млн";
    if (a >= 1e4) return num(v / 1e3, step >= 1e3 ? 0 : 1) + NBSP + "тыс.";
    var d = step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log10(step)));
    return num(v, d);
  }

  function pad(n) { return (n < 10 ? "0" : "") + n; }
  // Время показывается в часовом поясе данных: метки пришли как UTC-секунды
  // от наивных отметок, поэтому переводы часов браузера не сдвигают часы.
  function date(t) { var d = new Date(t * 1000); return pad(d.getUTCDate()) + "." + pad(d.getUTCMonth() + 1); }
  function dateFull(t) { var d = new Date(t * 1000); return date(t) + "." + d.getUTCFullYear(); }
  function hourSpan(t) {
    var d = new Date(t * 1000), h = d.getUTCHours();
    return DAYS[d.getUTCDay()] + " " + date(t) + ", " + pad(h) + "–" + pad((h + 1) % 24) + " ч";
  }
  function month(t) { var d = new Date(t * 1000); return MONTHS[d.getUTCMonth()] + " " + d.getUTCFullYear(); }
  function monthOnly(t) { return MONTHS[new Date(t * 1000).getUTCMonth()]; }

  window.fmt = { num: num, sig3: sig3, rub: rub, axis: axis, date: date, dateFull: dateFull,
                 hourSpan: hourSpan, month: month, monthOnly: monthOnly, MINUS: MINUS, NBSP: NBSP };
})();
