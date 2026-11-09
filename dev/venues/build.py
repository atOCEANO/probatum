"""Write the venue presets in python/probatum/venues from exchange-router's instrument
lists and margin tiers, read on the day it runs, which each file records as the day its
tables were taken. It asks the venues through exchange-router 4.0.0, which the
Contributor Guide's run line installs, so it runs by hand and never in the gate.

The instruments, each one's tick, lot and tier table, come from the router alone, which
serves no tiers for Binance's perpetuals, so they have no preset. The fee rates, the
liquidation models and Hyperliquid's price and maintenance rules are the venues'
documented ones, given here.
"""

import datetime
import json
import math
import pathlib

import exchange_router

OUT   = pathlib.Path(__file__).resolve().parents[2] / "python" / "probatum" / "venues"
TODAY = datetime.date.today().isoformat()

# the realistic option in each row of the fill model, a starting calibration
FILLS = {
    "limit":     "through",
    "liquidity": {"budget": 0.1},
    "spread":    "estimated",
    "slippage":  {"fixed": 1.0},
    "impact":    {"sqrt": 0.1},
    "sizes":     "on",
}


def continuous(rows):
    """Tier rows of (up_to, leverage, rate) with each amount continuity sets, the
    requirement meeting itself at every bound."""
    out, amount = [], 0.0
    for k, (up_to, leverage, rate) in enumerate(rows):
        if k:
            amount += (rate - rows[k - 1][2]) * rows[k - 1][0]
        out.append((up_to, leverage, rate, amount))

    return out


def listed(router, exchange, market):
    """Every instrument the router lists for ``exchange`` on ``market``, by its symbol,
    with the full information the router gives for it."""
    markets = router.get_markets(exchange=exchange, market_type=market)
    return {entry["symbol"]: router.get_symbol_info(exchange=exchange,
                                                    market_type=market,
                                                    symbol=entry["symbol"])
            for entry in markets}


def grid(info):
    """An instrument's tick and lot as the router gives them, each left out where the
    router gives none, which the preset reads as no rule."""
    steps = {"tick": info.price_tick, "lot": info.qty_step}
    return {name: step for name, step in steps.items() if step is not None}


def known(number, absent):
    """``number``, or ``absent`` where the router left it NaN."""
    return absent if math.isnan(number) else number


def tiers(router, exchange, market, symbol):
    """The router's tier table for ``symbol`` as rows of (up_to, leverage, rate,
    amount), an open last row's bound infinite and an absent deduction nothing."""
    table = router.get_margin_tiers(exchange=exchange, market_type=market,
                                    symbol=symbol)
    return [(known(row.up_to, math.inf), row.max_leverage, row.maintenance_margin_rate,
             known(row.maintenance_deduction, 0.0)) for row in table.itertuples()]


def spot(router, exchange):
    """``exchange``'s spot pairs as the router lists them, each with its tick and its
    lot."""
    pairs = listed(router, exchange, "spot")
    return {symbol: grid(info) for symbol, info in pairs.items()}


def bybit(router, market):
    """Bybit's perpetuals of ``market``, linear or inverse, as the router lists them,
    each with its tick, its lot, in contracts on an inverse contract, and its risk
    limits, by notional in the coin there."""
    out = {}
    for symbol, info in listed(router, "bybit", market).items():
        rows = tiers(router, "bybit", market, symbol)
        if rows:
            out[symbol] = {**grid(info), "tiers": rows}

    return out


def okx(router, market):
    """OKX's swaps of ``market``, linear or inverse, as the router lists them, each with
    its tick, its lot and its cross-margin position tiers by size, in the coin on a
    linear swap and in contracts on an inverse one."""
    out = {}
    for symbol, info in listed(router, "okx", market).items():
        rows = tiers(router, "okx", market, symbol)
        if rows:
            out[symbol] = {**grid(info), "tiers_by": "quantity", "tiers": rows}

    return out


def hyperliquid(router):
    """Hyperliquid's perpetuals as the router lists them, the builder deployed dexes'
    among them: each one's lot, its cap on decimals, and its margin table, maintenance
    at half the initial rate of each tier's leverage and the last tier open."""
    out = {}
    for symbol, info in listed(router, "hyperliquid", "linear").items():
        rows = tiers(router, "hyperliquid", "linear", symbol)
        if not rows:
            continue

        given = [(up_to, leverage, 0.5 / leverage) for up_to, leverage, _, _ in rows]
        out[symbol] = {
            "lot":      info.qty_step,
            "decimals": info.price_decimals,
            "tiers":    continuous(given),
        }

    return out


def number(value):
    """A float as TOML writes it."""
    return "inf" if value == float("inf") else repr(float(value))


def value(item):
    """``item`` as a TOML value."""
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, (int, float)):
        return number(item) if isinstance(item, float) else str(item)
    if isinstance(item, str):
        return json.dumps(item)
    if isinstance(item, list):
        return "[" + ", ".join(value(each) for each in item) + "]"
    if isinstance(item, dict):
        return "{ " + ", ".join(f"{key} = {value(v)}" for key, v in item.items()) + " }"
    raise TypeError(item)


def key(name):
    """``name`` as a TOML key, quoted where it is not a bare one, with every character
    past printable ASCII escaped by its code point, which TOML reads for any of them,
    where a JSON escape splits one past the first plane in two."""
    if name.isascii() and name.replace("_", "").replace("-", "").isalnum():
        return name

    quoted = (c if " " <= c <= "~" and c not in '"\\' else f"\\U{ord(c):08X}"
              for c in name)
    return '"' + "".join(quoted) + '"'


def save(name, lines):
    """The preset ``name`` written to its file as ``lines``."""
    head = ["# written by dev/venues/build.py, which a rerun rewrites", "", "[venue]",
            f"name = {value(name)}", f"taken = {value(TODAY)}"]
    (OUT / f"{name}.toml").write_text("\n".join(head + lines) + "\n", encoding="utf-8",
                                      newline="\n")


def write(name, kind, defaults, instruments=None):
    """The preset ``name`` of instruments of ``kind`` written to its file."""
    lines = [f"kind = {value(kind)}", "", "[defaults]"]
    lines += [f"{setting} = {value(v)}" for setting, v in defaults.items()]
    lines += ["", "[defaults.fills]"]
    lines += [f"{setting} = {value(v)}" for setting, v in FILLS.items()]

    for symbol, spec in sorted((instruments or {}).items()):
        lines += ["", f"[instruments.{key(symbol)}]"]
        for setting, v in spec.items():
            if setting != "tiers":
                lines.append(f"{setting} = {value(v)}")
                continue

            # an amount of 0 is the loader's own, and a quantity table's every one
            lines.append("tiers = [")
            lines += [f"    {{ up_to = {number(up_to)}, leverage = {number(leverage)}, "
                      f"rate = {number(rate)}"
                      + (f", amount = {number(amount)} }}," if amount else " },")
                      for up_to, leverage, rate, amount in v]
            lines.append("]")

    save(name, lines)
    print(name, len(instruments or {}))


def combine(name, parts):
    """The preset ``name`` combining the presets ``parts`` in one account, written to
    its file."""
    save(name, [f"combines = {value(parts)}"])
    print(name, "combines", ", ".join(parts))


def main():
    perp   = dict(liquidation="takeover", funding="data")
    venues = ["binance", "bybit", "okx", "hyperliquid"]

    with exchange_router.Router.local(exchanges=venues) as router:
        write("binance_spot", "spot", dict(fees={"maker": 0.001, "taker": 0.001},
                                           funding="none"),
              spot(router, "binance"))

        write("bybit_perp", "linear",
              dict(fees={"maker": 0.0002, "taker": 0.00055}, **perp),
              bybit(router, "linear"))
        write("bybit_inverse", "inverse",
              dict(fees={"maker": 0.0002, "taker": 0.00055}, **perp),
              bybit(router, "inverse"))
        write("bybit_spot", "spot", dict(fees={"maker": 0.001, "taker": 0.001},
                                         funding="none"),
              spot(router, "bybit"))
        combine("bybit", ["bybit_spot", "bybit_perp"])

        write("okx_perp", "linear",
              dict(fees={"maker": 0.0002, "taker": 0.0005}, **perp),
              okx(router, "linear"))
        write("okx_inverse", "inverse",
              dict(fees={"maker": 0.0002, "taker": 0.0005}, **perp),
              okx(router, "inverse"))
        write("okx_spot", "spot", dict(fees={"maker": 0.0008, "taker": 0.001},
                                       funding="none"),
              spot(router, "okx"))
        combine("okx", ["okx_spot", "okx_perp"])

        write("hyperliquid_perp", "linear",
              dict(fees={"maker": 0.00015, "taker": 0.00045},
                   liquidation="book_then_backstop", backstop=2.0 / 3.0, funding="data",
                   sigfigs=5),
              hyperliquid(router))


if __name__ == "__main__":
    main()
