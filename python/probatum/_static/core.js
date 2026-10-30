"use strict";

// spec to chart. Everything drawn is a key in the document, and nothing this file
// decides is a colour or a number taken off the data, which is what lets a Python
// test in a gate with no browser assert what a chart shows. What it does decide is
// what only the renderer knows: the grain a given width of axis can carry, the
// locale and the stamp a number is written in, and the type scale in device
// pixels (ADR 0042)

let SPEC = null;
let THEME = null;
let MODE = "dark";
let chart = null;
let anchors = [];                      // one series per panel, the coordinate frame
let redraws = [];                      // primitive invalidators
let cursor = 0;                        // bar under the crosshair
let selection = null;                  // half open [from, to, fillIndex] of the selected trade
let UI = 1;

const LWC = LightweightCharts;
const FONT = '"JetBrains Mono", ui-monospace, "SF Mono", Menlo, Consolas, monospace';

const STYLE = { solid: 0, dotted: 1, dashed: 2 };
const SCALE = { linear: 0, log: 1, percent: 2 };

// the palette keys chart.css reads. Written on mount and on every toggle, so the
// stylesheet and the chart cannot drift apart
const CSS_VARS = {
  plane: "--plane", surface: "--surface", ink: "--ink", ink2: "--ink-2",
  muted: "--muted", hairline: "--hairline", win: "--win", loss: "--loss",
  s1: "--s1",
};

const T = function () {
  return THEME[MODE];
};

// "auto" is answered here and not in Python, and that is not a hole in ADR 0042:
// the choice is between two palettes Python already shipped, on an input Python
// was not in the room to see

// the notebook's own theme where the chart can see the page around it: JupyterLab
// marks its body light or dark and VS Code names its theme in a class; a page of
// another origin keeps both to itself, and then the reader's system decides
const hostMode = function () {
  try {
    if (window.parent === window) return null;
    const body = window.parent.document.body;
    const light = body.getAttribute("data-jp-theme-light");
    if (light === "true") return "light";
    if (light === "false") return "dark";
    if (body.classList.contains("vscode-light")) return "light";
    if (body.classList.contains("vscode-dark")
        || body.classList.contains("vscode-high-contrast")) return "dark";
  } catch (error) {
    return null;
  }
  return null;
};

const resolveMode = function (mode) {
  if (mode !== "auto") return mode;
  const host = hostMode();
  if (host) return host;
  const ask = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)");
  return ask && ask.matches ? "dark" : "light";
};

const invalidate = function () {
  redraws.forEach(function (f) { f(); });
};

// one sub-linear scale factor for everything: a chart three times wider does not
// want three times bigger type. Both dimensions feed it and the tighter one wins,
// since a notebook cell is wide and short and several panes stacked into 400px
// leave the labels eating the plot. The floor is 1.0 because nothing should ever
// render below the size it was authored at, least of all in the cramped cell
// where legibility is already worst
const scaleFor = function (w, h) {
  const byW = 0.92 + w / 3400;
  const byH = 0.88 + h / 1700;
  return Math.max(1.0, Math.min(1.5, Math.min(byW, byH)));
};

// an unstyled line takes the next slot in the theme's series palette. Every one
// of them used to fall through to s1, so the flagship chart drew EMA 20 and EMA
// 60 in one blue with two identical legend swatches, and a third line would have
// been the same blue again. The slot is stored rather than the colour it resolves
// to, so a theme toggle re-reads it.
const SERIES_SLOTS = 4;

const slotColor = function (spec) {
  if (spec.color) return spec.color;
  if (spec.slot === undefined) return T().s1;
  return T()["s" + (spec.slot + 1)] || T().s1;
};

const fmt = function (v, d) {
  return (v === null || v === undefined || Number.isNaN(v)) ? "n/a"
    : v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
};

const stamp = function (i) {
  return new Date(SPEC.t[i] * 1000).toISOString().slice(0, 16).replace("T", " ");
};


// -------------------------------------------------------------------- the axis
//
// The renderer weighs every tick on its own and will straddle two of its own
// thresholds, so a strip 1920 wide read 9, 17, 12:00, Sept, 9, 17, Oct: one
// intraday tick between two day numbers, which is noise rather than detail. What
// is drawn here is only which ticks to drop; answering null keeps the renderer's
// own label, and it holds every one this file has no argument with.

const DAY = 86400;

// generous, because one tick at a time cannot tell a lone fine tick from a sub-grid
const coarsestTick = function (seconds) {
  if (seconds > DAY * 100) return 1;       // months and years
  if (seconds > DAY * 21) return 2;        // and days
  return 4;                                // and times
};

let axisSpan = 0;                      // seconds on screen
let axisFrom = null;                   // and no label before this one
let axisEdge = null;                   // no label past this time; it would be cut

const tickMark = function (time, kind) {
  if (kind > coarsestTick(axisSpan)) return "";
  // a label is centred on its bar and cut by the edge of the plot, so a year of
  // hourly candles ended the axis on "20" and opened it on "25". The room has to
  // come out of the gutter and nothing here can buy it: rightOffset is priced in
  // bars, which narrows every candle to pay for one label. Dropping it is the
  // trade, since the date is on the crosshair and in the legend either way, and
  // both ends because they are one defect seen twice
  if (axisEdge !== null && time > axisEdge) return "";
  if (axisFrom !== null && time < axisFrom) return "";
  return null;
};

// measured on a range change rather than inside the formatter, because the
// formatter is called from the middle of the axis layout and asking the time
// scale for its own geometry from in there is a re-entry waiting to happen
const measureAxis = function (range) {
  if (!range || !SPEC.t.length) return;
  const last = SPEC.t.length - 1;
  const clamp = function (v) { return Math.max(0, Math.min(last, v)); };
  const span = SPEC.t[clamp(Math.ceil(range.to))] - SPEC.t[clamp(Math.floor(range.from))];

  const plot = document.getElementById("chart").clientWidth -
    chart.priceScale("right").width();
  const spacing = plot / Math.max(1, range.to - range.from);
  // half a four digit year at the axis font, plus air, expressed in bars. The
  // axis is set at 12 * UI, where "2026" measures about 26 * UI wide, so half of
  // it is 13. Erring wide costs a label that would have fitted and erring narrow
  // puts a clipped one back, and the whole point here is not to clip
  const room = Math.ceil((16 * UI) / Math.max(spacing, 0.0001));
  const edge = SPEC.t[clamp(Math.ceil(range.to) - room)];
  // clamped, so a chart panned to leave slack on the left suppresses nothing:
  // the label only clips when bar zero is against the edge
  const from = SPEC.t[clamp(Math.floor(range.from) + room)];

  // guarded rather than unconditional: re-applying marks the axis dirty, and an
  // unguarded write would repaint on every frame of a drag forever
  if (span === axisSpan && edge === axisEdge && from === axisFrom) return;
  axisSpan = span;
  axisEdge = edge;
  axisFrom = from;
  chart.timeScale().applyOptions({ tickMarkFormatter: tickMark });
};

// a resize moves no bars, so it fires no range change, and the rule above is in
// pixels: a chart dragged narrower would keep suppressing for the width it used
// to have. The guard inside measureAxis is what makes asking again cheap
const remeasureAxis = function () {
  measureAxis(chart.timeScale().getVisibleLogicalRange());
};

// a Track carries i0 and a value array, so the alignment contract arrives as an
// integer and this file never learns the rule. A null becomes whitespace, {time}
// with no value, rather than a dropped row, because dropping would make the
// neighbours adjacent. That is necessary and it is not sufficient: the renderer
// joins straight across whitespace too, so the hole is cut in `paint` by splitting
// the track into one series per run (ADR 0042)
const track = function (tr) {
  if (!tr) return [];
  const out = new Array(tr.v.length);
  for (let j = 0; j < tr.v.length; j++) {
    const v = tr.v[j];
    const time = SPEC.t[tr.i0 + j];
    out[j] = v === null ? { time: time } : { time: time, value: v };
  }
  return out;
};

// palette lookup. p[0] is always null, so an absent colour needs no branch here
// and no sentinel on the wire
const colorAt = function (ca, i) {
  if (!ca) return null;
  const j = i - ca.i0;
  if (j < 0 || j >= ca.k.length) return null;
  return ca.p[ca.k[j]];
};

const isRamp = function (c) {
  return c !== null && c !== undefined && typeof c === "object";
};

// the lowest value drawn on a pane, read once: its candles' lows, its volume, the
// engine's curve on it, and every series and level placed on it
const LOWEST = [];
const lowest = function (index) {
  if (LOWEST[index] === undefined) {
    const panel = SPEC.panels[index];
    let low = Infinity;
    const take = function (v) { if (typeof v === "number" && v < low) low = v; };
    if (panel.candles) SPEC.ohlc.forEach(function (o) { if (o !== null) take(o[2]); });
    if (panel.volume && SPEC.vol) SPEC.vol.v.forEach(take);
    if (panel.name === "equity" && SPEC.equity) SPEC.equity.v.forEach(take);
    if (panel.name === "drawdown" && SPEC.drawdown) SPEC.drawdown.v.forEach(take);
    SPEC.series.forEach(function (s) {
      if (s.panel !== panel.name) return;
      [s.v, s.v2, s.values].forEach(function (vs) { if (vs) vs.forEach(take); });
      if (s.kind === "level") take(s.value);
    });
    LOWEST[index] = low;
  }
  return LOWEST[index];
};

// a tick inside the range a pane was pinned to, not under zero where nothing on the
// pane goes below it, and clear of every level's label
const ticked = function (index, price) {
  const panel = SPEC.panels[index];
  if (panel.range) {
    const slack = 1e-9 * Math.abs(panel.range[1] - panel.range[0]);
    if (price < panel.range[0] - slack || price > panel.range[1] + slack) return false;
  }
  if (price < 0 && lowest(index) >= 0) return false;
  const frame = frameFor(index);
  const y = frame.priceToCoordinate(price);
  if (y === null) return true;
  const near = Math.round(12 * UI) + 4;
  return !SPEC.series.some(function (s) {
    if (s.kind !== "level" || s.panel !== panel.name) return false;
    const at = frame.priceToCoordinate(s.value);
    return at !== null && Math.abs(at - y) < near;
  });
};

const priceFormat = function (digits, unit, index) {
  // thousands apart, as every other number on the page is
  const format = function (price) { return fmt(price, digits) + (unit || ""); };
  const made = { type: "custom", minMove: Math.pow(10, -digits), formatter: format };
  if (index !== undefined) {
    made.tickmarksFormatter = function (prices) {
      return prices.map(function (price) {
        return ticked(index, price) ? format(price) : "";
      });
    };
  }
  return made;
};

// a pinned range is honoured by every series on that panel rather than by the
// panel: v5 has no setVisibleRange on a price scale, and autoscaleInfoProvider is
// only consulted for a series that carries data
const pinned = function (panel) {
  if (!panel.range) return {};
  const lo = panel.range[0], hi = panel.range[1];
  return {
    autoscaleInfoProvider: function () {
      return { priceRange: { minValue: lo, maxValue: hi } };
    },
  };
};

const panelIndex = function (name) {
  for (let i = 0; i < SPEC.panels.length; i++) {
    if (SPEC.panels[i].name === name) return i;
  }
  return 0;
};


// ------------------------------------------------------------------ series

// the coordinate frame for a panel: an invisible line covering every bar, so the
// pane exists in panel order even before anything is drawn on it, levels have
// something to hang off, and a primitive has a price scale to convert against
const addAnchor = function (panel, index) {
  const s = chart.addSeries(LWC.LineSeries, Object.assign({
    color: "rgba(0,0,0,0)", lineWidth: 1,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
    priceFormat: priceFormat(panel.digits, panel.name === "drawdown" ? "%" : "",
                             index),
  }, pinned(panel)), index);
  s.setData(SPEC.t.map(function (time) { return { time: time }; }));
  return s;
};

const addCandles = function (panel, index) {
  const s = chart.addSeries(SPEC.candles.style === "bars" ? LWC.BarSeries
    : SPEC.candles.style === "line" ? LWC.LineSeries : LWC.CandlestickSeries,
    Object.assign({
    // the one series in the file that did not turn its badge off, and it read as
    // the omission it was. It covers an axis label rather than replacing one,
    // because the last price does not land on a gridline, and it says nothing
    // new: at rest the cursor is the last bar, so the legend's own C is that
    // number already
    borderVisible: true, priceLineVisible: false, lastValueVisible: false,
    priceFormat: priceFormat(panel.digits),
  }, pinned(panel)), index);
  return s;
};

const addLine = function (spec, panel, index) {
  const shared = Object.assign({
    lineWidth: spec.width,
    lineStyle: STYLE[spec.style],
    lineType: spec.step ? LWC.LineType.WithSteps : LWC.LineType.Simple,
    priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
    priceFormat: priceFormat(spec.digits),
  }, pinned(panel));

  // a fill under a line is an area, which is a different series type rather than
  // an option on this one. Stops run bottom to top like every other fill, so the
  // first sits on the panel floor and the last touches the line. An area carries
  // one lineColor and ignores a per-point colour entirely, so a ramp here was
  // painted flat while the legend went on reporting a colour per bar; that pair is
  // refused at construction rather than resolved in favour of one half
  if (spec.fill) {
    const stops = spec.fill;
    return chart.addSeries(LWC.AreaSeries, Object.assign({
      lineColor: slotColor(spec),
      bottomColor: stops[0],
      topColor: stops[stops.length - 1],
    }, shared), index);
  }

  // `undefined` is not "no opinion", it is the vendored bundle's opinion: its
  // merge skips an undefined key, so the library's own default blue survives and
  // a bar the ramp left uncoloured is painted in it while the legend reports it
  // grey. That is the colour rule of ADR 0042 leaking out through a hole the
  // asset grep cannot see, since the literal lives in the vendored bundle rather
  // than in ours
  return chart.addSeries(LWC.LineSeries, Object.assign({
    color: isRamp(spec.color) ? T().muted : slotColor(spec),
  }, shared), index);
};

const addHistogram = function (spec, panel, index) {
  const s = chart.addSeries(LWC.HistogramSeries, Object.assign({
    // not undefined: see addLine. The bundle's own default would show through
    color: isRamp(spec.color) ? T().muted : (spec.color || T().s4),
    base: spec.base,
    priceLineVisible: false, lastValueVisible: false,
    priceFormat: priceFormat(spec.digits),
  }, pinned(panel)), index);
  return s;
};

// the runs of real values in a row list, split at the whitespace. A run is what
// can be drawn as one continuous line; the holes between them are the gaps
const runsOf = function (rows) {
  const runs = [];
  let open = null;
  for (let j = 0; j < rows.length; j++) {
    if (rows[j].value === undefined) {
      open = null;
      continue;
    }
    if (open === null) {
      open = [];
      runs.push(open);
    }
    open.push(rows[j]);
  }
  return runs;
};

// per-item colour is data, so it is written into the point rather than into the
// series options, and it has to be rewritten on a theme change.
//
// The one series per contiguous run below is not in the spec because the spec
// was never wrong: it is what it takes to paint a gap the renderer would
// otherwise join straight across
const paint = function (entry) {
  const spec = entry.spec;
  const rows = track(spec);
  if (isRamp(spec.color)) {
    for (let j = 0; j < rows.length; j++) {
      const c = colorAt(spec.color, spec.i0 + j);
      if (c !== null && rows[j].value !== undefined) rows[j].color = c;
    }
  }
  if (spec.equityBase !== undefined) {
    for (let j = 0; j < rows.length; j++) {
      if (rows[j].value !== undefined) {
        rows[j].color = rows[j].value >= spec.equityBase ? T().win : T().loss;
      }
    }
  }
  if (!entry.make) {
    entry.series.setData(rows);
    return;
  }
  const runs = runsOf(rows);
  if (!entry.siblings) entry.siblings = [];
  while (entry.siblings.length < runs.length - 1) entry.siblings.push(entry.make());
  const drawn = [entry.series].concat(entry.siblings);
  for (let k = 0; k < drawn.length; k++) drawn[k].setData(runs[k] || []);
};

const paintCandles = function () {
  const tint = SPEC.candles.color;
  const rows = new Array(SPEC.n);
  for (let i = 0; i < SPEC.n; i++) {
    const o = SPEC.ohlc[i];
    // a bar the instrument printed no bar on is whitespace, so its slot stays on
    // the axis and its neighbours keep their places
    if (o === null) {
      rows[i] = { time: SPEC.t[i] };
      continue;
    }
    const row = SPEC.candles.style === "line" ? { time: SPEC.t[i], value: o[3] }
      : { time: SPEC.t[i], open: o[0], high: o[1], low: o[2], close: o[3] };
    const c = colorAt(tint, i);
    if (c !== null) {
      row.borderColor = c;
      row.wickColor = c;
      row.color = o[3] >= o[0] && SPEC.candles.style === "hollow" ? "rgba(0,0,0,0)" : c;
    }
    rows[i] = row;
  }
  CANDLES.setData(rows);
};

const paintVolume = function () {
  if (!VOLUME) return;
  const rows = track(SPEC.vol);
  for (let j = 0; j < rows.length; j++) {
    const o = SPEC.ohlc[SPEC.vol.i0 + j];
    if (o !== null) rows[j].color = o[3] >= o[0] ? T().volume_up : T().volume_down;
  }
  VOLUME.setData(rows);
};

let CANDLES = null;
let VOLUME = null;
let SERIES = [];                       // {spec, series} for every value-carrying mark


// ------------------------------------------------------------------- theme

// Lucide's sun and moon, in the frame of the footer's other icons
const ICON = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
  'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" ' +
  'focusable="false">';
const THEME_ICONS = {
  light: ICON + '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/>' +
    '<path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/>' +
    '<path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/>' +
    '<path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/></svg>',
  dark: ICON + '<path d="M20.985 12.486a9 9 0 1 1-9.473-9.472c.405-.022.617.46.402' +
    '.803a6 6 0 0 0 8.268 8.268c.344-.215.825-.004.803.401"/></svg>',
};

const applyTheme = function (mode) {
  MODE = mode;
  const t = T();
  const root = document.documentElement;
  root.setAttribute("data-theme", mode);
  root.style.setProperty("color-scheme", mode);
  // the button shows the mode it will switch to, here, which is the only place that
  // knows the mode moved, so a chart that opens light never offers light
  const button = document.getElementById("theme");
  if (button) {
    const next = mode === "dark" ? "light" : "dark";
    button.innerHTML = THEME_ICONS[next];
    button.title = "Switch to the " + next + " theme";
    button.setAttribute("aria-label", button.title);
  }
  Object.keys(CSS_VARS).forEach(function (key) {
    if (t[key] !== undefined) root.style.setProperty(CSS_VARS[key], t[key]);
  });

  chart.applyOptions({
    layout: {
      background: { type: LWC.ColorType.Solid, color: t.surface },
      textColor: t.ink2,
      panes: { separatorColor: t.hairline, separatorHoverColor: t.grid },
    },
    grid: { vertLines: { color: t.grid }, horzLines: { color: t.grid } },
    crosshair: {
      vertLine: { color: t.muted, labelBackgroundColor: t.axis },
      horzLine: { color: t.muted, labelBackgroundColor: t.axis },
    },
    rightPriceScale: { borderColor: t.axis },
    timeScale: { borderColor: t.axis },
  });

  CANDLES.applyOptions({
    upColor: SPEC.candles.style === "hollow" ? "rgba(0,0,0,0)" : t.up, downColor: t.down,
    borderUpColor: t.up, borderDownColor: t.down,
    wickUpColor: t.wick_up, wickDownColor: t.wick_down,
    color: t.s1, lineWidth: 2,
  });

  paintCandles();
  paintVolume();
  SERIES.forEach(paint);
  if (typeof repaintTrades === "function") repaintTrades();
  invalidate();
};


// -------------------------------------------------------------- primitives

// a primitive converts prices through a series, and a series with only
// whitespace has no values for the scale to fit, so the anchor is the last
// resort rather than the first choice
const frameFor = function (index) {
  const panel = SPEC.panels[index];
  if (panel.candles && CANDLES) return CANDLES;
  if (panel.volume && VOLUME) return VOLUME;
  for (let k = 0; k < SERIES.length; k++) {
    if (SERIES[k].spec.panel === panel.name) return SERIES[k].series;
  }
  return anchors[index];
};

// give a panel's anchor the extent of whatever is drawn on it, when nothing else
// on that panel carries values, which is the panel that used to mount clean and
// paint nothing. The anchor is fully transparent, so seeding it costs no pixels
const seedAnchors = function () {
  SPEC.panels.forEach(function (panel, index) {
    if (panel.candles || panel.volume) return;
    const carried = SERIES.some(function (entry) { return entry.spec.panel === panel.name; });
    if (carried) return;

    const seen = [];
    const take = function (v) { if (typeof v === "number" && isFinite(v)) seen.push(v); };
    SPEC.series.forEach(function (s) {
      if (s.panel !== panel.name) return;
      if (s.kind === "level") take(s.value);
      if (s.kind === "marker" && s.values) s.values.forEach(take);
      if (s.kind === "band") {
        if (s.v) s.v.forEach(take);
        if (s.v2) s.v2.forEach(take);
      }
    });
    if (!seen.length) return;

    const lo = Math.min.apply(null, seen);
    const hi = Math.max.apply(null, seen);
    anchors[index].setData(SPEC.t.map(function (time) {
      return { time: time, value: lo };
    }));
    // a pinned panel already carries its own provider from `pinned`, and it wins
    if (!panel.range) {
      anchors[index].applyOptions({
        autoscaleInfoProvider: function () {
          return { priceRange: { minValue: lo, maxValue: hi } };
        },
      });
    }
  });
};

// the edge of each pane toward a separator, shaded faintly in ink
const edgePrimitive = function (index, count) {
  return {
    updateAllViews: function () {},
    attached: function (p) { redraws.push(p.requestUpdate); },
    paneViews: function () {
      return [{
        zOrder: function () { return "bottom"; },
        renderer: function () {
          return {
            draw: function (target) {
              target.useMediaCoordinateSpace(function (scope) {
                const ctx = scope.context;
                const w = scope.mediaSize.width, h = scope.mediaSize.height;
                const reach = Math.round(16 * UI);
                const shade = function (from, to) {
                  const g = ctx.createLinearGradient(0, from, 0, to);
                  g.addColorStop(0, T().edge);
                  g.addColorStop(1, T().edgeClear);
                  ctx.fillStyle = g;
                  ctx.fillRect(0, Math.min(from, to), w, reach);
                };
                if (index > 0) shade(0, reach);
                if (index < count - 1) shade(h, h - reach);
              });
            },
          };
        },
      }];
    },
  };
};

// band, background and marker are the kinds no native series can draw, so they
// arrive here rather than in the series dispatch
const mountPrimitives = function () {
  const panes = chart.panes();

  SPEC.series.forEach(function (s) {
    const index = panelIndex(s.panel);
    if (s.kind === "band") {
      panes[index].attachPrimitive(bandPrimitive(frameFor(index), s));
    } else if (s.kind === "background") {
      panes[index].attachPrimitive(spanFill(
        function () { return s.spans; },
        function () { return s.fills; }
      ));
    } else if (s.kind === "marker") {
      panes[index].attachPrimitive(calloutPrimitive(frameFor(index), s));
    }
  });

  panes.forEach(function (pane, i) {
    if (SPEC.panels[i]) pane.attachPrimitive(legendPrimitive(i));
    if (panes.length > 1) pane.attachPrimitive(edgePrimitive(i, panes.length));
  });
};


// ------------------------------------------------------------------- mount

// the narrowest a bar may be drawn, set from the plot's width and the bars it holds
// so that fitting frames every one; reset as the plot is resized
const floorSpacing = function (fallback) {
  const width = chart.timeScale().width() || 0.8 * (fallback || 0);
  const bars = SPEC.t.length + (chart.timeScale().options().rightOffset || 0);
  if (width > 0 && bars > 0) {
    chart.timeScale().applyOptions({ minBarSpacing: Math.min(0.04, 0.98 * width / bars) });
  }
};

// the view the call asked for, every bar or its focus, framed again as the time axis
// settles to its width, until the reader moves the view by hand. Fitting a projection
// left as many empty bars before the first as it projected, so a projection is framed
// from the first bar to its own last
let reframe = true;
const frameView = function () {
  floorSpacing();
  if (!reframe) return;
  if (SPEC.focus) {
    chart.timeScale().setVisibleLogicalRange({ from: SPEC.focus[0], to: SPEC.focus[1] });
  } else if (SPEC.t.length > SPEC.n) {
    chart.timeScale().setVisibleLogicalRange({ from: -0.5, to: SPEC.t.length - 0.5 });
  } else {
    chart.timeScale().fitContent();
  }
};

// the bar the legend reads while no pointer is over the plot: the last one in view
let pointed = false;
const resting = function () {
  const range = chart.timeScale().getVisibleLogicalRange();
  const last = SPEC.n - 1;
  return range ? Math.max(0, Math.min(last, Math.floor(range.to))) : last;
};

const mount = function (spec, root) {
  if (window.frameElement) document.documentElement.classList.add("embedded");
  SPEC = spec;
  THEME = spec.theme;
  MODE = resolveMode(spec.theme.mode);
  cursor = spec.focus
    ? Math.max(0, Math.min(spec.n - 1, Math.floor(spec.focus[1]))) : spec.n - 1;

  const t = T();
  chart = LWC.createChart(root, {
    autoSize: true,
    layout: {
      background: { type: LWC.ColorType.Solid, color: t.surface },
      textColor: t.ink2, fontFamily: spec.theme.font || FONT, attributionLogo: false,
      panes: { separatorColor: t.hairline, separatorHoverColor: t.grid, enableResize: true },
    },
    grid: { vertLines: { color: t.grid }, horzLines: { color: t.grid } },
    crosshair: {
      mode: LWC.CrosshairMode.Normal,
      vertLine: {
        color: t.muted, width: 1, style: LWC.LineStyle.Solid, labelBackgroundColor: t.axis,
      },
      horzLine: {
        color: t.muted, width: 1, style: LWC.LineStyle.Solid, labelBackgroundColor: t.axis,
      },
    },
    rightPriceScale: { borderColor: t.axis },
    // the renderer will not draw a bar narrower than half a pixel, and it does not
    // say so: fitContent on a year of hourly candles asks for 8760 bars, hits the
    // floor at 4380px of width, and quietly frames the last 2256 instead. The chart
    // then showed a different slice of history at every window size, and a walk
    // forward whose whole subject is five fitted stretches opened on the last two.
    // Refusing to fit is worse than a crowded bar, because the crowding is visible.
    //
    // And then the crowding is answered rather than lived with. Conflation is the
    // renderer's own and was off: below one device pixel of bar spacing it
    // aggregates to the column, which took a year of hourly candles from 105,186
    // fillRect calls to 13,206 and from a grey smear to a readable year. The
    // hazard is worth naming, because it is not visible from here: legend.js
    // reads SPEC by logical index while the candle under the pointer is now a
    // group of bars, so a render test walks the pointer across a 4000 bar chart
    // and holds the two to each other
    timeScale: {
      borderColor: t.axis, timeVisible: true, secondsVisible: false,
      minBarSpacing: 0.04,
      enableConflation: true,
      tickMarkFormatter: tickMark,
    },
  });

  // panes come into being in panel order, because the anchor for panel k is the
  // first series asking for pane k
  anchors = spec.panels.map(addAnchor);

  spec.panels.forEach(function (panel, index) {
    if (panel.candles) CANDLES = addCandles(panel, index);
    if (panel.volume) {
      VOLUME = chart.addSeries(LWC.HistogramSeries, Object.assign({
        priceFormat: { type: "volume" }, priceLineVisible: false, lastValueVisible: false,
      }, pinned(panel)), index);
    }
  });

  SERIES = [];

  // slots are handed out per panel, in argument order, over the lines that named
  // no colour. Per panel rather than per chart because two colours only have to
  // tell each other apart inside the pane a reader is looking at, and counting
  // across the whole chart makes a panel's only series depend on how many other
  // panels came before it: the raw engine notebook draws its equity as an
  // ordinary Line on a panel it names equity, and a chart-wide count handed that
  // one curve the third colour while the same curve on the search notebook, where
  // it arrives as the engine's own track, stayed the first
  const taken = {};
  spec.series.forEach(function (s) {
    if (s.kind !== "line" || s.color) return;
    const n = taken[s.panel] || 0;
    s.slot = n % SERIES_SLOTS;
    taken[s.panel] = n + 1;
  });

  // the drawn series first, so a level can hang on one. A price line attached
  // to a series carrying only whitespace does not render, and the anchor is
  // exactly that: every Level in the library was reaching the renderer
  // correctly and drawing nothing at all
  spec.series.forEach(function (s) {
    const index = panelIndex(s.panel);
    const panel = spec.panels[index];
    if (s.kind === "line") {
      // the factory is what lets a gapped line grow a sibling per run in paint;
      // a histogram needs none, since its bars are already separate marks
      const make = function () { return addLine(s, panel, index); };
      SERIES.push({ spec: s, series: make(), make: make, siblings: [] });
    } else if (s.kind === "histogram") {
      SERIES.push({ spec: s, series: addHistogram(s, panel, index) });
    }
  });

  // the engine's own two curves. They carry a `make` like any other line, because
  // the gap split in `paint` only runs for an entry that has one: both are dense
  // by construction today, so this is a guard rather than a fix, and the reason to
  // write it is that the next curve added here will not be
  const addCurve = function (name, label, track, options, type) {
    const index = panelIndex(name);
    const panel = spec.panels[index];
    // by name, not by index: panelIndex answers 0 for a name it does not hold, so
    // an absent panel resolved to the price panel, which is truthy, and the guard
    // below let it through. The drawdown track ships even when its fall is shaded
    // onto the equity panel rather than given one of its own, so its curve was
    // added to the candles: a series running -22 to 0 on a scale running 95k to
    // 212k, with invertFilledArea flooding everything above it
    if (!panel || panel.name !== name) return;
    const unit = options.unit, base = options.base;
    delete options.unit;
    delete options.base;
    const make = function () {
      return chart.addSeries(type || LWC.AreaSeries, Object.assign({
        priceLineVisible: false, lastValueVisible: false,
        priceFormat: priceFormat(panel.digits, unit),
      }, options, pinned(panel)), index);
    };
    SERIES.push({
      spec: {
        kind: "line", panel: name, name: label, color: options.lineColor,
        equityBase: base,
        i0: track.i0, v: track.v, digits: panel.digits, unit: unit, own: true,
      },
      series: make(),
      make: make,
      siblings: [],
    });
  };

  if (spec.equity) {
    addCurve("equity", "equity", spec.equity, {
      base: spec.equity.v[0], color: t.win, lineWidth: 2,
    }, LWC.LineSeries);
  }
  if (spec.drawdown) {
    addCurve("drawdown", "drawdown", spec.drawdown, {
      unit: "%",
      lineColor: t.loss, topColor: t.ddTop, bottomColor: t.ddBottom,
      lineWidth: 1, invertFilledArea: true,
    });
  }

  // after the two curves above, not before them: the engine's own panels get
  // their series only there, and a Level placed before them fell through to the
  // whitespace anchor and drew nothing
  spec.series.forEach(function (s) {
    if (s.kind !== "level") return;
    const index = panelIndex(s.panel);
    // muted, not axis: the axis colour is chosen to sit almost invisibly
    // against the plane, which is right for a gridline and wrong for a
    // reference the reader is meant to see
    frameFor(index).createPriceLine({
      price: s.value,
      color: s.color || T().muted,
      lineWidth: s.width,
      lineStyle: STYLE[s.style],
      axisLabelVisible: true,
      title: s.name || "",
    });
  });

  chart.panes().forEach(function (pane, i) {
    if (spec.panels[i]) pane.setStretchFactor(spec.panels[i].weight);
    const scale = pane.priceScale("right");
    if (scale) scale.applyOptions({ mode: SCALE[spec.panels[i].scale], entireTextOnly: true });
  });

  paintCandles();
  paintVolume();
  SERIES.forEach(paint);

  // the cursor travels the whole axis rather than the candles, because with
  // future= the axis is longer and a series drawn past the last candle still has
  // values to read. legend.js is what knows a bar past the candles has no OHLC
  chart.subscribeCrosshairMove(function (p) {
    const last = spec.t.length - 1;
    const i = (p && p.logical !== null && p.logical !== undefined)
      ? Math.max(0, Math.min(last, p.logical)) : resting();
    pointed = Boolean(p && p.logical !== null && p.logical !== undefined);
    if (i !== cursor) { cursor = i; invalidate(); }
  });

  // before anything frames the chart, so the first paint is already measured
  // rather than corrected a frame later
  chart.timeScale().subscribeVisibleLogicalRangeChange(measureAxis);

  // before the primitives, because each of them converts through a series and
  // this is what gives a primitive-only panel one that carries values
  seedAnchors();

  mountPrimitives();
  if (typeof mountTrades === "function") mountTrades();
  if (typeof mountControls === "function") mountControls();

  applyTheme(MODE);

  floorSpacing(root.clientWidth);
  ["pointerdown", "wheel", "touchstart", "keydown"].forEach(function (name) {
    document.addEventListener(name, function () { reframe = false; },
                              { capture: true, passive: true });
  });
  chart.timeScale().subscribeSizeChange(frameView);
  if (spec.focus) {
    chart.timeScale().setVisibleLogicalRange({ from: spec.focus[0], to: spec.focus[1] });
  } else {
    // room reserved for the projection BEFORE fitting, because fitContent fits to
    // the data and the projected times carry whitespace: a chart asking for
    // `future=` whose projection is drawn by a primitive, or by markers, or by
    // any series that does not itself carry values out there, framed to the last
    // real bar and left the projection off the right edge with nothing on screen
    // saying it existed. The condition is whether a NATIVE series reaches those
    // times, not which mark drew them, so a T-long line beside a projected band
    // reproduces it too
    const ahead = spec.t.length - spec.n;
    if (ahead > 0) chart.timeScale().applyOptions({ rightOffset: ahead });
    floorSpacing(root.clientWidth);
    frameView();
  }
};
