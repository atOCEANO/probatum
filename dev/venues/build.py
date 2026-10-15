"""Write the venue presets in python/probatum/venues from each venue's own tables,
read from its public endpoints on the day it runs, which each file records as the day
its tables were taken. It asks the venues themselves, so it runs by hand and never in
the gate:

    docker run --rm -v "$PWD:/src" -w /src python:3.13-slim python dev/venues/build.py

The instruments, each one's tick, lot and tier table, come from the endpoints; the fee
rates, the liquidation models and Hyperliquid's price rule are the venues' documented
ones, given here.
"""

import datetime
import decimal
import json
import pathlib
import time
import urllib.parse
import urllib.request

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

BINANCE           = "https://www.binance.com/bapi/futures/v1/friendly"
BINANCE_LINEAR    = "https://fapi.binance.com/fapi/v1/exchangeInfo"
BINANCE_INVERSE   = "https://dapi.binance.com/dapi/v1/exchangeInfo"
BYBIT_INSTRUMENTS = "https://api.bybit.com/v5/market/instruments-info"
BYBIT_TIERS       = "https://api.bybit.com/v5/market/risk-limit"
OKX_INSTRUMENTS   = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"
OKX_TIERS         = "https://www.okx.com/api/v5/public/position-tiers"
HYPERLIQUID       = "https://api.hyperliquid.xyz/info"


def fetch(url, body=None):
    """The JSON at ``url``, posted ``body`` where given, tried three times."""
    data    = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data,
                                     method="GET" if body is None else "POST",
                                     headers={"Content-Type": "application/json",
                                              "User-Agent": "probatum-presets"})

    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2.0 * (attempt + 1))


def printed(*texts):
    """The product of numbers as a venue prints them, taken in decimal, so a step
    times a contract's size is the float nearest the number the venue means."""
    out = decimal.Decimal(1)
    for text in texts:
        out *= decimal.Decimal(text)

    return float(out)


def continuous(rows):
    """Tier rows of (up_to, leverage, rate) with each amount continuity sets, the
    requirement meeting itself at every bound."""
    out, amount = [], 0.0
    for k, (up_to, leverage, rate) in enumerate(rows):
        if k:
            amount += (rate - rows[k - 1][2]) * rows[k - 1][0]
        out.append((up_to, leverage, rate, amount))

    return out


def binance(market):
    """Binance's perpetuals of ``market``, linear or inverse, as the router names them:
    each one trading, its tick, its lot, in contracts on an inverse contract, and
    its leverage brackets, notional in the coin there."""
    info     = fetch(BINANCE_LINEAR if market == "linear" else BINANCE_INVERSE)
    book     = "future" if market == "linear" else "delivery"
    tiers    = fetch(f"{BINANCE}/{book}/common/brackets", {})
    brackets = {entry["symbol"]: entry["riskBrackets"]
                for entry in tiers["data"]["brackets"]}

    out = {}
    for symbol in info["symbols"]:
        status = symbol.get("status", symbol.get("contractStatus"))
        if symbol["contractType"] != "PERPETUAL" or status != "TRADING":
            continue

        rows = brackets.get(symbol["symbol"])
        if not rows:
            continue

        rows  = sorted(rows, key=lambda row: row["bracketNotionalCap"])
        rules = {f["filterType"]: f for f in symbol["filters"]}
        out[symbol["symbol"].removesuffix("_PERP")] = {
            "tick":  printed(rules["PRICE_FILTER"]["tickSize"]),
            "lot":   printed(rules["LOT_SIZE"]["stepSize"]),
            "tiers": [(float(row["bracketNotionalCap"]),
                       float(row["maxOpenPosLeverage"]),
                       float(row["bracketMaintenanceMarginRate"]),
                       float(row["cumFastMaintenanceAmount"])) for row in rows],
        }

    return out


def bybit(category):
    """Bybit's perpetuals of ``category``, linear or inverse, as the router names them:
    each crypto one trading, its tick, its lot, in contracts on an inverse
    contract, its risk limits, and its own rates in the Innovation Zone."""
    def pages(url):
        rows, cursor = [], ""
        while True:
            page = fetch(f"{url}?category={category}&limit=1000"
                         + (f"&cursor={urllib.parse.quote(cursor)}" if cursor else ""))
            rows += page["result"]["list"]
            cursor = page["result"].get("nextPageCursor")
            if not cursor:
                return rows

    wanted      = "LinearPerpetual" if category == "linear" else "InversePerpetual"
    instruments = {row["symbol"]: row for row in pages(BYBIT_INSTRUMENTS)
                   if row["contractType"] == wanted and row["status"] == "Trading"
                   and row.get("symbolType", "") in ("", "innovation")}

    limits = {}
    for row in pages(BYBIT_TIERS):
        if row["symbol"] in instruments:
            limits.setdefault(row["symbol"], []).append(row)

    out = {}
    for symbol, row in instruments.items():
        if symbol not in limits:
            continue

        tiers = sorted(limits[symbol], key=lambda tier: float(tier["riskLimitValue"]))
        # a first tier deducts nothing, which Bybit leaves blank
        spec = {"tick": printed(row["priceFilter"]["tickSize"]),
                "lot": printed(row["lotSizeFilter"]["qtyStep"]),
                "tiers": [(float(t["riskLimitValue"]), float(t["maxLeverage"]),
                           float(t["maintenanceMargin"]),
                           float(t["mmDeduction"] or 0.0)) for t in tiers]}
        if row.get("symbolType") == "innovation":
            spec["fees"] = {"maker": 0.0004, "taker": 0.0011}
        out[symbol] = spec

    return out


def okx(kind):
    """OKX's swaps of ``kind``, linear or inverse, as the router names them: each crypto
    one live and out of pre-market, its tick, its lot and its cross-margin position
    tiers by size, in the base coin on a linear swap and in contracts on an inverse
    one."""
    swaps = [swap for swap in fetch(OKX_INSTRUMENTS)["data"]
             if swap["ctType"] == kind and swap["state"] == "live"
             and swap["instCategory"] == "1" and swap["ruleType"] == "normal"]

    families = {}
    for family in sorted({swap["instFamily"] for swap in swaps}):
        time.sleep(0.25)
        query = urllib.parse.urlencode(dict(instType="SWAP", tdMode="cross",
                                            instFamily=family))
        families[family] = fetch(f"{OKX_TIERS}?{query}")["data"]

    out = {}
    for swap in swaps:
        rows = sorted(families.get(swap["instFamily"]) or [],
                      key=lambda tier: float(tier["maxSz"]))
        if not rows:
            continue

        per = swap["ctVal"] if kind == "linear" else "1"
        out[swap["instId"].removesuffix("-SWAP").replace("-", "")] = {
            "tick": printed(swap["tickSz"]), "lot": printed(swap["lotSz"], per),
            "tiers_by": "quantity",
            "tiers": [(printed(t["maxSz"], per), float(t["maxLever"]), float(t["mmr"]),
                       0.0) for t in rows]}

    return out


def hyperliquid():
    """Hyperliquid's perpetuals as the router names them, upper-cased: each one listed,
    its margin table, maintenance at half the initial rate of each tier's leverage and
    the last tier open, its lot, one unit of the last of its size decimals, and its
    cap on decimals, six less them."""
    meta   = fetch(HYPERLIQUID, {"type": "meta"})
    tables = {key: table["marginTiers"] for key, table in meta["marginTables"]}
    out    = {}

    for asset in meta["universe"]:
        if asset.get("isDelisted"):
            continue

        table = asset["marginTableId"]
        # an id below 50 is a single tier with that leverage
        tiers = ([{"lowerBound": "0.0", "maxLeverage": table}] if table < 50
                 else tables[table])
        bounds = [float(tier["lowerBound"]) for tier in tiers[1:]] + [float("inf")]
        given  = [(bound, float(tier["maxLeverage"]), 0.5 / float(tier["maxLeverage"]))
                  for bound, tier in zip(bounds, tiers)]
        out[f"{asset['name'].upper()}USDC"] = {
            "lot":      printed(f"1e-{asset['szDecimals']}"),
            "decimals": 6 - asset["szDecimals"],
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
    perp = dict(liquidation="takeover", funding="data")

    write("binance_perp", "linear",
          dict(fees={"maker": 0.0002, "taker": 0.0005}, reduce_only_exempt=True,
               **perp),
          binance("linear"))
    write("binance_inverse", "inverse",
          dict(fees={"maker": 0.0002, "taker": 0.0005}, **perp), binance("inverse"))
    write("binance_spot", "spot", dict(fees={"maker": 0.001, "taker": 0.001},
                                       funding="none"))
    combine("binance", ["binance_spot", "binance_perp"])

    write("bybit_perp", "linear",
          dict(fees={"maker": 0.0002, "taker": 0.00055}, **perp), bybit("linear"))
    write("bybit_inverse", "inverse",
          dict(fees={"maker": 0.0002, "taker": 0.00055}, **perp), bybit("inverse"))
    write("bybit_spot", "spot", dict(fees={"maker": 0.001, "taker": 0.001},
                                     funding="none"))
    combine("bybit", ["bybit_spot", "bybit_perp"])

    write("okx_perp", "linear",
          dict(fees={"maker": 0.0002, "taker": 0.0005}, **perp), okx("linear"))
    write("okx_inverse", "inverse",
          dict(fees={"maker": 0.0002, "taker": 0.0005}, **perp), okx("inverse"))
    write("okx_spot", "spot", dict(fees={"maker": 0.0008, "taker": 0.001},
                                   funding="none"))
    combine("okx", ["okx_spot", "okx_perp"])

    write("hyperliquid_perp", "linear",
          dict(fees={"maker": 0.00015, "taker": 0.00045},
               liquidation="book_then_backstop", backstop=2.0 / 3.0, funding="data",
               sigfigs=5),
          hyperliquid())


if __name__ == "__main__":
    main()
