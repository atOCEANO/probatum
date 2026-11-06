<h1>OCEΛNO <small><code>probatum</code></small></h1>


<div style="padding-top: 0px;">
  <a href="https://github.com/atOCEANO/probatum/releases"><img src="https://img.shields.io/github/v/release/atOCEANO/probatum?label=release&color=2ea043" alt="Latest release" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+" /></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.88-orange.svg?logo=rust&logoColor=white" alt="Rust 1.88" /></a>
  <a href="https://gymnasium.farama.org/"><img src="https://img.shields.io/badge/Gymnasium-1.2+-0081a5.svg" alt="Gymnasium 1.2+" /></a>
  <a href="https://github.com/atOCEANO/probatum/releases"><img src="https://img.shields.io/badge/wheels-Linux%20%7C%20macOS%20%7C%20Windows-informational.svg" alt="Wheels for Linux, macOS and Windows" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
</div>

<sub>
  <a href="../README.md">Introduction</a> &nbsp;•&nbsp;
  <a href="Research_Guide.md">Research Guide</a> &nbsp;•&nbsp;
  <a href="RL_Guide.md">RL Guide</a> &nbsp;•&nbsp;
  <a href="Charts.md">Charts</a> &nbsp;•&nbsp;
  <a href="Python_API.md">Python API</a> &nbsp;•&nbsp;
  <a href="Specification.md">Specification</a> &nbsp;•&nbsp;
  <b>Venue Notes</b> &nbsp;•&nbsp;
  <a href="Validation_Guide.md">Validation Guide</a> &nbsp;•&nbsp;
  <a href="Decisions.md">Decisions</a> &nbsp;•&nbsp;
  <a href="Migration.md">Migration</a> &nbsp;•&nbsp;
  <a href="Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <a href="Contributor_Guide.md">Contributor Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Venue Notes

Each preset is a TOML file in `python/probatum/venues/`, read as a `probatum.Venue` the first time it is asked for. `dev/venues/build.py` writes every preset but `frictionless`. It takes the instruments, and each one's price rule, lot and margin tiers, from exchange-router ([0048](Decisions.md)), Binance's tiers from the table behind Binance's website, and the fee rates, the forced close and Hyperliquid's price and maintenance rules as the venue documents them. Every file the script writes records the day it ran as `taken`, the day its tables describe ([0028](Decisions.md)).

```python
import probatum

perp = probatum.venues.binance_perp

print(perp.name, perp.taken, perp.kind)  # its name, the day its tables were taken, its kind
print(perp.describe())                   # what a run here assumes, graded
```

A preset is data, so it ages: a venue lists and delists instruments, moves its tiers and changes its rates, and nothing checks a file against the live venue. A rerun of the script rewrites the presets, and its diff is what the venues changed since.

<br>
<br>

## Shared Conventions

### Symbols

Every instrument is keyed by the name exchange-router gives it, so a Data from `Data.from_router` finds its tables with no mapping:
* Binance's and Bybit's own symbols, with Binance's `_PERP` suffix dropped from an inverse contract (`BTCUSD`);
* OKX's instrument id without its dashes and its `-SWAP` (`BTC-USDT-SWAP` is `BTCUSDT`);
* and Hyperliquid's coin upper-cased with its quote, `USDC` (`kPEPE` is `KPEPEUSDC`), led on a builder-deployed dex by the dex's name (`xyz:TSLA` is `XYZTSLAUSDC`).

In a preset that combines a venue's markets, each key carries its kind, `spot:BTCUSDT` and `linear:BTCUSDT`.

### Instruments

A preset lists the instruments exchange-router lists as trading on the day it was taken, each with its tick or price rule and its lot, and a perpetual with its margin tiers. A symbol it does not list has no lot, price rule or tiers, so with the preset's sizes on, a run refuses it by name; `replace(instruments={...})` gives it its own, or `replace(fills={"sizes": "off"})` runs it without, a perpetual then capped at the venue's `leverage` and keeping no maintenance. `hyperliquid_perp` refuses an unlisted perpetual whatever its sizes, since its price rule needs the instrument's decimals and its forced close the instrument's tiers, and `replace(instruments={...})` gives it both. What each venue's presets hold:
* Every instrument the router lists, whatever its asset: Binance's TradFi perpetuals and tokenized stock pairs, Bybit's stock, ETF, commodity and forex perpetuals and its xStocks pairs, OKX's stock and commodity swaps and pairs, its pre-market swaps among them, and the perpetuals of Hyperliquid's builder-deployed dexes.
* No preset holds a dated future.

### Fees

`fees` holds the venue's base maker and taker rates, each a share of a fill's notional, as charged with no trading volume behind the account and no discount for holding the venue's token. Every instrument pays them except where its own table says otherwise. No preset holds the venue's volume tiers or a maker rebate, since the rate an account earns depends on its volume across the whole venue.

No preset holds the rates a venue charges a group of its instruments on a schedule of their own either. Each venue's section below names its groups, and every instrument in one pays the preset's base rates until `replace(instruments={...})` gives it its own.

On a preset of one kind, `replace(fees={"maker": ..., "taker": ...})` sets an account's own rates. On `binance`, `bybit` and `okx` each kind's table holds its part's rates, so the rates go there, `replace(kinds={"spot": {"fees": ...}, "linear": {"fees": ...}})`, and an instrument with rates of its own keeps them.

### Margin tiers

A perpetual's `tiers` are the venue's rows of maximum leverage and maintenance as a position grows ([0015](Decisions.md)):
* `up_to`, the size a row reaches;
* `leverage`, the most a position there may be levered;
* `rate`, its maintenance rate;
* and `amount`, the deduction that makes the requirement continuous at each bound.

A row's size is a notional in the quote currency where `tiers_by` is `notional`, in the coin on an inverse contract, and a quantity where it is `quantity`, as OKX states its tiers. The last row may be open, its bound infinite.

### Funding

Every perpetual preset charges `funding = "data"`: the funding events the Data carries, at their own times and rates, charged on the positions held then ([Specification](Specification.md#funding)). `Data.from_router(..., funding=True)` fetches them, and a run on bars that carry none for a perpetual is refused rather than charged nothing. A spot preset charges none.

### The forced close

Binance, Bybit and OKX take an account over where its margin runs out and book what is left as the liquidation charge. Hyperliquid sells every leg on the book from its trigger and leaves the trader the rest, falling back to the takeover where the sale would leave less than two thirds of the maintenance requirement ([0019](Decisions.md), [Specification](Specification.md#liquidation)).

### The fill model

Every preset but `frictionless` fills through the same starting calibration, the realistic option in each row of the fill model ([0013](Decisions.md)):

| Setting | Value | What it does |
| :--- | :--- | :--- |
| `limit` | `"through"` | a limit fills once the price trades half a step of its price rule past it |
| `liquidity` | `{"budget": 0.1}` | a bar's orders share a tenth of the trailing volume |
| `spread` | `"estimated"` | a taker pays half the spread the trailing bars give |
| `slippage` | `{"fixed": 1.0}` | and one basis point beside it |
| `impact` | `{"sqrt": 0.1}` | and square-root impact in its size against the trailing volume |
| `sizes` | `"on"` | every order rounded to its instrument's lot and price rule |
| `window` | `24` | the bars before a fill that the spread, impact and budget read |

These are a starting point rather than a measurement of anyone's costs. `probatum.cost_curve` and `probatum.breakeven_bps` show how much a result depends on them ([Research Guide](Research_Guide.md#costs-and-capacity)).

### What no preset holds

* **The router's order limits.** An instrument's minimum notional and its largest order come with its bars from `Data.from_router`, as of the fetch, where no table of the preset gives one.
* **Delisting times.** A preset lists what was trading when it was taken. An instrument's `delist`, which closes its position at its last close and refuses every order on it from then on, is given by hand in `instruments`.
* **Volume tiers, rebates, token discounts and a group's own rates.** See [Fees](#fees).
* **Isolated margin, price bands and auto-deleveraging.** The account is margined across its perpetuals, and a forced close is the venue's own model above.

<br>
<br>

## Binance

`binance_spot`, `binance_perp` and `binance_inverse` hold every pair or perpetual exchange-router lists for Binance, the perpetuals with their leverage brackets. `binance` combines the spot and linear presets in one account of two wallets.

| Preset | Maker | Taker | Tiers from |
| :--- | :--- | :--- | :--- |
| `binance_spot` | 0.10% | 0.10% | |
| `binance_perp` | 0.02% | 0.05% | each symbol's leverage brackets, by notional in the quote currency |
| `binance_inverse` | 0.02% | 0.05% | each symbol's leverage brackets, by notional in the coin |

An instrument's tick is its price filter's and its lot its lot-size filter's, in the coin on a linear perpetual and in contracts on an inverse one. `binance_perp` spares a reduce-only order the minimum notional. Binance charges its TradFi perpetuals, which its exchange information marks with the contract type `TRADIFI_PERPETUAL`, on a fee schedule of their own.

<br>

## Bybit

`bybit_spot`, `bybit_perp` and `bybit_inverse` hold every pair or perpetual exchange-router lists for Bybit, the perpetuals with their risk limits. `bybit` combines the spot and linear presets in one account.

| Preset | Maker | Taker | Tiers from |
| :--- | :--- | :--- | :--- |
| `bybit_spot` | 0.10% | 0.10% | |
| `bybit_perp` | 0.02% | 0.055% | each symbol's risk limits, by notional in the quote currency, USDT or USDC |
| `bybit_inverse` | 0.02% | 0.055% | each symbol's risk limits, by notional in the coin |

A spot pair's lot is its base coin's precision. Bybit charges a perpetual in its Innovation Zone, which its instrument list marks with the symbol type `innovation`, 0.04% maker and 0.11% taker.

<br>

## OKX

`okx_spot`, `okx_perp` and `okx_inverse` hold every pair or swap exchange-router lists for OKX, the swaps with their cross-margin position tiers. `okx` combines the spot and linear presets in one account.

| Preset | Maker | Taker | Tiers from |
| :--- | :--- | :--- | :--- |
| `okx_spot` | 0.08% | 0.10% | |
| `okx_perp` | 0.02% | 0.05% | each family's position tiers, by size in the coin |
| `okx_inverse` | 0.02% | 0.05% | each family's position tiers, by size in contracts |

OKX's tiers are by position size and carry no deduction, so every row's `amount` is zero. A linear swap's lot and tier sizes are the venue's, in contracts, times the contract's value in the coin.

<br>

## Hyperliquid

`hyperliquid_perp` holds every perpetual exchange-router lists for Hyperliquid, those of its builder-deployed dexes among them.

| Preset | Maker | Taker | Tiers from |
| :--- | :--- | :--- | :--- |
| `hyperliquid_perp` | 0.015% | 0.045% | each asset's margin table, by notional in USDC |

Hyperliquid's prices have no tick. A price is held to five significant figures, a whole number always allowed, and to at most six decimals less the asset's size decimals. A size is held to its size decimals, so an asset with two size decimals trades in lots of 0.01. Its margin tables give a maximum leverage for each tier, and maintenance is half the initial rate that leverage implies, each tier's deduction set by continuity and the last tier open; an asset whose table is a single leverage has one tier. A market on a builder-deployed dex adds to the base rates the fee share its deployer sets for it, and a market in growth mode has the whole fee cut by 90% or more; the dex's metadata gives each market's `deployerFeeScale` and `growthMode`. The forced close sells on the book first, with the backstop at two thirds of the requirement, as [the forced close](#the-forced-close) says.

<br>

## frictionless

`frictionless` charges nothing: no fees, no funding, no spread, slippage or impact, a budget of the whole trailing volume, limits filled on touch and no size rules. It runs any instrument of any market, for teaching and for checking a rule's logic apart from its costs. It is not the baseline that separates an edge from what a venue took: a Result's baseline without trading costs keeps the run's own venue and removes only what trading charges ([Python API](Python_API.md#costs-assumptions-and-baselines)).
