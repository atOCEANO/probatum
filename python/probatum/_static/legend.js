"use strict";

// legend: one per pane, painted on canvas rather than built as DOM. Canvas needs
// no layout space, and pane separators are draggable, which would leave a DOM
// overlay needing repositioning on a drag that fires no event.
//
// Every panel prints its own name first, the candles' panel the symbol it draws. A
// panel is created the first time a panel= mentions it, so a typo makes a stray
// panel, and showing the name is what turns that into something you cannot miss on
// screen.

const legendValueAt = function (spec, i) {
  const j = i - spec.i0;
  if (j < 0 || j >= spec.v.length) return null;
  return spec.v[j];
};

// a gap is an absence, and the shared formatter calls it "n/a". That is right in
// a table cell, where a column has to line up and a blank one reads as data that
// went missing, and it is wrong hovering over the plot: `trailing stop  n/a`
// reads as a fault in the chart rather than as the bars where there was no stop.
// The row keeps its swatch and its name and stops quoting a number, which is what
// the line beside it is doing
const legendValue = function (v, digits) {
  return (v === null || v === undefined || Number.isNaN(v)) ? undefined : fmt(v, digits);
};

// a background is the only mark that says what a stretch of bars IS, and it had
// no way of saying which stretch: three shaded regimes and no key, so the reader
// got calm, normal and wild as three washes and a guess. Only a named one draws a
// row, because shading used as scene should not be made to explain itself.
//
// The swatch takes the bottom stop, which is the strongest, since a fill runs
// bottom to top. Outlined, because a background wash is a smudge against the
// plane rather than a colour
const backgroundRow = function (spec, i) {
  for (let k = 0; k < spec.spans.length; k++) {
    const span = spec.spans[k];
    if (i >= span[0] && i < span[1]) {
      const stops = spec.fills[span[2]];
      return {
        swatch: stops[0], outline: true, label: spec.name,
        value: spec.keys ? spec.keys[span[2]] : undefined,
      };
    }
  }
  // outside every region, and saying so by having nothing to show rather than by
  // naming the absence
  return { label: spec.name };
};

const united = function (value, unit) {
  return value === undefined || !unit ? value : value + unit;
};

const bandRow = function (spec, i, digits) {
  const j = i - spec.i0;
  const a = legendValueAt(spec, i);
  const b = spec.v2 ? (j >= 0 && j < spec.v2.length ? spec.v2[j] : null) : spec.level;
  const both = [a, b].every(function (v) { return typeof v === "number" && isFinite(v); });
  const edge = isRamp(spec.color) ? colorAt(spec.color, i) : spec.color;
  const fill = Array.isArray(spec.fill) ? spec.fill[spec.fill.length - 1]
    : isRamp(spec.fill) ? colorAt(spec.fill, i) : null;
  return {
    swatch: edge || fill || T().muted, outline: !edge, label: spec.name,
    value: both ? fmt(Math.min(a, b), digits) + " to " + fmt(Math.max(a, b), digits)
      : undefined,
  };
};

// a band whose two edges are named lines on its pane, whose rows give both numbers; a
// band starts where both its edges do, so a line may start before it
const edged = function (band) {
  if (band.edged === undefined) {
    const drawn = function (v) {
      return SPEC.series.some(function (s) {
        if (s.kind !== "line" || s.panel !== band.panel || !s.name) return false;
        const shift = band.i0 - s.i0;
        return shift >= 0 && shift + v.length <= s.v.length
          && v.every(function (x, j) { return x === s.v[shift + j]; });
      });
    };
    band.edged = Boolean(band.v2) && drawn(band.v) && drawn(band.v2);
  }
  return band.edged;
};

const legendRows = function (index) {
  const panel = SPEC.panels[index];
  const i = cursor;
  const rows = [{ label: panel.candles && SPEC.symbol ? SPEC.symbol : panel.name,
                  strong: true }];
  const held = chart.panes()[index] && chart.panes()[index].priceScale("right").options();
  if (held && held.mode === 1) rows.push({ label: "log" });
  if (held && held.mode === 2) rows.push({ label: "%" });
  if (held && !held.autoScale) rows.push({ label: "fixed" });

  // a bar past the last candle exists on the axis but has no OHLC, because
  // future= lengthens the axis and never invents a bar that traded, and a bar the
  // instrument printed no bar on has none either
  const bar = i < SPEC.n ? SPEC.ohlc[i] : null;

  if (panel.candles) {
    if (SPEC.interval) rows.push({ label: SPEC.interval });
    rows.push({ label: stamp(i) });
    if (bar !== null) {
      const up = bar[3] >= bar[0];
      rows.push({
        swatch: up ? T().up : T().down, hollow: up && SPEC.candles.style === "hollow",
        label: "O", value: fmt(bar[0], panel.digits),
      });
      rows.push({ label: "H", value: fmt(bar[1], panel.digits) });
      rows.push({ label: "L", value: fmt(bar[2], panel.digits) });
      rows.push({ label: "C", value: fmt(bar[3], panel.digits) });
    } else {
      rows.push({ label: i < SPEC.n ? "no bar" : "projected" });
    }
  }

  if (panel.volume && SPEC.vol && bar !== null) {
    rows.push({
      swatch: bar[3] >= bar[0] ? T().volume_up : T().volume_down,
      label: "", value: legendValue(legendValueAt(SPEC.vol, i), panel.digits),
    });
  }

  // before the drawn series, because a regime is the ground everything else on
  // the panel is standing on
  SPEC.series.forEach(function (s) {
    if (s.kind !== "background" || s.panel !== panel.name || !s.name) return;
    rows.push(backgroundRow(s, i));
  });

  SPEC.series.forEach(function (s) {
    if (s.kind !== "band" || s.panel !== panel.name || !s.name) return;
    if (panel.name === "equity" && s.name === "drawdown") return;
    if (edged(s)) return;
    rows.push(bandRow(s, i, panel.digits));
  });

  SERIES.forEach(function (entry) {
    const spec = entry.spec;
    if (spec.panel !== panel.name) return;
    const at = spec.equityBase === undefined ? null : legendValueAt(spec, i);
    const swatch = spec.equityBase !== undefined
      ? (at === null || at >= spec.equityBase ? T().win : T().loss)
      : isRamp(spec.color) ? colorAt(spec.color, i) : slotColor(spec);
    rows.push({
      swatch: swatch || T().muted,
      label: spec.own ? "" : spec.name || "",
      value: united(legendValue(legendValueAt(spec, i), spec.digits), spec.unit),
    });
  });

  if (panel.name === "equity" && SPEC.equity) {
    const v = legendValueAt(SPEC.equity, i);
    const base = SPEC.equity.v[0];
    if (v !== null && base) {
      const r = (v / base - 1) * 100;
      rows.push({ label: "ret", value: (r >= 0 ? "+" : "") + fmt(r, 2) + "%" });
    }
    // shaded on this panel rather than given a pane of its own, the fall has no
    // series to report it, so the number is read straight off the track and put
    // where the shading is. A pane of its own still carries its own row
    const own = SPEC.panels.some(function (p) { return p.name === "drawdown"; });
    if (SPEC.drawdown && !own) {
      const d = legendValueAt(SPEC.drawdown, i);
      if (d !== null) rows.push({ label: "drawdown", value: fmt(d, 2) + "%" });
    }
  }

  return rows;
};

// the share of a pane its top margin may take: what its legend took at rest, but never
// so much that under 60 pixels of plot are left above the bottom margin, where the
// arrows over and under the bars would widen the scale until the candles flattened
const marginCap = function (h) {
  return Math.max(0.15, Math.min(0.5, (0.88 * h - 60 * UI) / h));
};

// the room each pane's legend took at rest, which controls.js sets the pane's top margin
// from; it only grows at one width and type scale, so a margin it moves cannot move it
// back, and either changing starts it again
let legendKey = "";
let legendRest = [];
const rested = function (index, room) {
  if (pointed || legendRest[index] >= room) return;
  legendRest[index] = room;
  requestAnimationFrame(placePaneCtls);
};

const legendPrimitive = function (index) {
  return {
    updateAllViews: function () {},
    attached: function (p) { redraws.push(p.requestUpdate); },
    paneViews: function () {
      return [{
        zOrder: function () { return "top"; },
        renderer: function () {
          return {
            draw: function (target) {
              target.useMediaCoordinateSpace(function (scope) {
                const ctx = scope.context, t = T();
                // a pane too short to hold a label without covering its own data
                // shows none; the value is still on the crosshair axis label.
                //
                // Deliberately low, and it stays low: raising it to a multiple of
                // the label's own height stopped a short pane defacing its curve
                // and also silenced a deliberately crowded one, which had been
                // drawing its legend and counting the series it could not fit. A
                // legend that drops one series and says so is the tested
                // behaviour; a pane that draws none at all says nothing
                if (scope.mediaSize.height < 26 * UI) return;

                const fs = Math.round(12.5 * UI);
                const pad = Math.round(11 * UI), gap = Math.round(6 * UI);
                const swS = Math.round(fs * 0.6), swGap = Math.round(5 * UI);
                const sepW = Math.round(15 * UI), lineH = Math.round(fs * 1.5);
                const maxX = scope.mediaSize.width - Math.round(6 * UI);
                // canvas text does not wrap, so it is done by hand: an entry that
                // does not fit moves to a second line, and if the pane is too
                // short for one the remainder is counted rather than dropped
                // up to a third of the pane, or two lines where they take no
                // more than half of it; the pane's margin follows the lines
                const h = scope.mediaSize.height;
                const share = function (part) { return Math.floor((h * part - pad) / lineH); };
                // a pane one line of legend would take half of is left to its plot at
                // rest; the crosshair still reads it
                if (!pointed && pad + lineH > h / 2) return;

                ctx.save();
                ctx.textBaseline = "alphabetic";

                const measure = function (r) {
                  ctx.font = (r.strong ? "600 " : "400 ") + fs + "px " + FONT;
                  let w = ctx.measureText(r.label).width;
                  if (r.swatch) w += swS + swGap;
                  if (r.value !== null && r.value !== undefined) {
                    ctx.font = "400 " + fs + "px " + FONT;
                    w += (r.label ? gap : 0) + ctx.measureText(String(r.value)).width;
                  }
                  return w;
                };

                const paintRow = function (r, x, y) {
                  if (r.swatch) {
                    if (r.hollow) {
                      ctx.strokeStyle = r.swatch;
                      ctx.lineWidth = Math.max(1, Math.round(1.5 * UI));
                      ctx.strokeRect(x + 0.5, y - swS + 0.5, swS - 1, swS - 1);
                    } else {
                      ctx.fillStyle = r.swatch;
                      ctx.fillRect(x, y - swS, swS, swS);
                      // a background's fill is a wash, so its square needs an
                      // edge to be findable without the edge lying about what is
                      // inside it
                      if (r.outline) {
                        ctx.strokeStyle = t.muted;
                        ctx.lineWidth = 1;
                        ctx.strokeRect(x + 0.5, y - swS + 0.5, swS - 1, swS - 1);
                      }
                    }
                    x += swS + swGap;
                  }
                  if (r.label) {
                    ctx.font = (r.strong ? "600 " : "400 ") + fs + "px " + FONT;
                    ctx.fillStyle = r.strong ? t.ink : t.muted;
                    ctx.fillText(r.label, x, y);
                    x += ctx.measureText(r.label).width + gap;
                  }
                  if (r.value !== null && r.value !== undefined) {
                    ctx.font = "400 " + fs + "px " + FONT;
                    ctx.fillStyle = t.ink2;
                    ctx.fillText(String(r.value), x, y);
                  }
                };

                const rows = legendRows(index);
                // a pane's one series is named on a second line rather than counted,
                // where the margin's cap leaves room for it
                const maxLines = Math.max(1, Math.min(share(marginCap(h)), rows.length <= 2 ? 2
                  : Math.max(share(1 / 3), Math.min(2, share(0.5)))));
                // the counter has to fit too, or the count vanishes exactly when
                // it is the only thing left saying something is missing
                const more = function (left) { return "+" + left + " more"; };
                ctx.font = "600 " + fs + "px " + FONT;
                const moreW = ctx.measureText(more(rows.length)).width + sepW;

                let line = 0, x = pad, lastX = pad;
                for (let k = 0; k < rows.length; k++) {
                  const w = measure(rows[k]);
                  const last = line + 1 >= maxLines;
                  // `x > pad` used to guard this, which let the FIRST entry on a
                  // line paint however wide it was: canvas clipped it at the pane
                  // edge and no counter was drawn, so a narrow panel dropped a
                  // series in silence. That is the exact failure this block is
                  // here to prevent, produced by the block itself
                  const overflows = x + w > (last && k < rows.length - 1 ? maxX - moreW : maxX);
                  if (overflows) {
                    if (last) {
                      ctx.font = "600 " + fs + "px " + FONT;
                      ctx.fillStyle = t.muted;
                      ctx.fillText(more(rows.length - k), lastX, pad + fs * 0.85 + line * lineH);
                      break;
                    }
                    line++; x = pad;
                    // still too wide on a fresh line, and this is now the last
                    // one: report the remainder rather than clipping it
                    if (line + 1 >= maxLines
                        && x + w > (k < rows.length - 1 ? maxX - moreW : maxX)) {
                      ctx.font = "600 " + fs + "px " + FONT;
                      ctx.fillStyle = t.muted;
                      ctx.fillText(more(rows.length - k), x, pad + fs * 0.85 + line * lineH);
                      break;
                    }
                  }
                  paintRow(rows[k], x, pad + fs * 0.85 + line * lineH);
                  x += w + sepW;
                  lastX = x;
                }
                rested(index, pad + (line + 1) * lineH);
                ctx.restore();
              });
            },
          };
        },
      }];
    },
  };
};
