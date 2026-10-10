<h1>OCEΛNO <small><code>embeddable-market-simulation-library</code></small></h1>


<div style="padding-top: 0px;">
  <a href="https://github.com/atOCEANO/embeddable-market-simulation-library/releases"><img src="https://img.shields.io/github/v/release/atOCEANO/embeddable-market-simulation-library?label=release&color=2ea043" alt="Latest release" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.13+-blue.svg" alt="Python 3.13+" /></a>
  <a href="https://www.rust-lang.org/"><img src="https://img.shields.io/badge/rust-1.88-orange.svg?logo=rust&logoColor=white" alt="Rust 1.88" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" /></a>
</div>

<sub>
  <a href="../README.md">Introduction</a> &nbsp;•&nbsp;
  <a href="Python_API.md">Python API</a> &nbsp;•&nbsp;
  <b>Specification</b> &nbsp;•&nbsp;
  <a href="RL_Guide.md">RL Guide</a> &nbsp;•&nbsp;
  <a href="Plotting.md">Plotting</a> &nbsp;•&nbsp;
  <a href="Architecture.md">Architecture</a> &nbsp;•&nbsp;
  <a href="Decisions.md">Decisions</a> &nbsp;•&nbsp;
  <a href="Contributor_Guide.md">Contributor Guide</a> &nbsp;•&nbsp;
  <a href="Validation_Guide.md">Validation Guide</a>
</sub>

<br>
<br>
<br>
<br>

## Specification

This page is every rule the engine applies to money: when an order fills and at what price, what each fill costs, when funding is charged, how margin is measured, where an account is liquidated, and what a run records. Each rule is stated as an equation where it has one and followed by a worked example, and every worked example is a test, so a number on this page is a number the engine produces. Why each rule is the way it is lives in [Decisions](Decisions.md).

The contracts are spot, linear perpetuals margined in the quote currency, and inverse perpetuals margined in their coin. Products are written with `*`, and every price is in the quote currency unless a rule says otherwise.

<br>

### Notation

For each linear perpetual `i` held in the margin wallet:

| Symbol | Meaning |
| :--- | :--- |
| `q_i` | the signed position in base units, positive long |
| `e_i` | its average entry price |
| `p_i` | a price |
| `W` | the margin wallet's cash |
| `N_i = abs(q_i) * p_i` | the notional |
| `E = W + sum_i q_i * (p_i - e_i)` | margin equity |
| `L_k` | the maximum leverage of tier `k` |
| `m_k`, `c_k` | the maintenance rate and cumulative amount of tier `k` |
| `f` | a fee rate: maker, taker or liquidation |

Inverse perpetuals count in their own quantities ([Inverse contracts](#inverse-contracts)).

<br>
<br>


## Time

### The clock

- A Data's clock is the open times, in UTC nanoseconds, at which any of its instruments printed a bar, in order. Nothing is invented between them.
- Every intraday row is a whole number of intervals from the first, so two instruments whose bars open at different minutes are refused. A daily or longer bar is placed at 00:00 UTC on the date it carries in its own time zone.
- An instrument is unavailable on a row where it printed no bar: its five values are NaN, it fills nothing, its market and resting orders wait for its next bar, and its position keeps its last mark.
- An instrument enters at its first bar. At its delisting time its waiting orders are cancelled and its position is closed at its last close with no fee, with the reason `delist`.

**Worked example.** A stock's daily bars on Thursday, Friday and Monday are three consecutive rows. A 3-bar average on Monday averages Thursday, Friday and Monday; a market order decided at Friday's close fills at Monday's open; and a stop that Monday's open gaps through fills at that open.

**Worked example of a delisting.** An instrument delisted at 12:00, whose last hourly bar opened at 11:00 and closed at 50.00, while another instrument goes on printing. A long of 2 held into 12:00 closes on the 12:00 row at 50.00 with no fee and the reason `delist`, and a market order still waiting on it is cancelled.

### Annualisation

`periods_per_year = (T - 1) * 365 / D`, where `T` is the number of rows on the clock and `D` the days from the first row's open to the last row's. It is taken once over the whole Data, so every slice of it annualises alike, and a Data needs two rows to have one.

**Worked example.** Hourly bars from 00:00 on 1 January of a year that is not a leap year to 00:00 on 1 January of the next are 8,761 rows over 365 days: `8,760 * 365 / 365 = 8,760`. A stock's 253 daily bars from one 2 January to the next annualise at 252.

### Coarser bars and outside series

- `resample(interval)` needs a whole multiple of the Data's interval. A coarse bar takes the first open, the highest high, the lowest low and the last close of its bars and the sum of their volumes, and is available if any of them was. Coarse bars align to 00:00 UTC and weeks to Monday, shifted by the Data's own offset when its bars open off the hour. A coarse bar the data does not cover to its end is dropped.
- `asof(values, when)` puts on row `t` the last value known by the end of row `t`, and NaN before the first. A value from a coarser Data is known at the end of its bar.

**Worked example.** Hourly bars, and 4-hour bars that open at 00:00 and 04:00. The 00:00 bar ends at 04:00, as the hourly bar that opens at 03:00 does, so its value first appears on the 03:00 row; the 04:00 to 06:00 rows keep it, and the 04:00 bar's value arrives on the 07:00 row.

<br>
<br>


## The step

### The order of events

From bar `t` to bar `t + 1`, in this order:

1. Transfers decided at `t` are applied ([Wallets](#wallets)).
2. Funding stamped after bar `t`'s end and up to bar `t + 1`'s open is charged at the last marks ([Funding](#funding)).
3. The liquidation path is solved for the positions carried in ([Liquidation](#liquidation)): the account is liquidated before the bar opens, or every held perpetual's bar is clipped at the clip point, or nothing happens.
4. Pending market orders fill at the open of the clipped bars, in the order they were placed.
5. Resting limits and stops resolve against the clipped bars in slot order. The exits of the position each fill meets, as the fills before it left that position, are reordered by the PnL per unit each would realize at its fill price, a stop's slippage included, worst first under the pessimistic setting, among the slots those exits hold, so a position the bar's own fills open has its exits ordered as a carried one's are; exits on different instruments are never reordered. Every reducing fill passes the margin bound.
6. Under the pessimistic setting, the perpetual fills that would reduce the position they meet, priced on the favourable side of where that position started on the bar, are held back when the path without them still reaches the trigger.
7. The liquidation check runs along the path for the account as it now stands.
8. Funding owned by the bar is charged at the closes on the positions still held.
9. Every position is marked at its close, and the ledger row is written.

An order decided at the close of bar `t` never fills on bar `t`.

### What a bar cannot show

A bar gives four prices and no order between them. The venue's `intrabar` setting decides every event that order would settle, and `"pessimistic"` is the default for every venue and every preset.

| Event | `"pessimistic"` | `"optimistic"` |
| :--- | :--- | :--- |
| a stop and a take-profit of one position both reached | the worse exit fills first | the better exit fills first |
| a take-profit and the liquidation trigger both reached | the take-profit is held back | the take-profit fills first |
| a funding event stamped inside the bar | a debit counts toward the check and a credit does not | a credit counts toward it and a debit does not |
| several instruments, each reaching its worst price | they reach them at the same moment | each reaches it while the others sit at their best |

`"random"` draws one of the two for each event from the stream named `intrabar` in the run's seed, which in a batch is named with the environment's index as well. Every bar draws in one order whether or not each event arises: the instruments together, the exits held back, each instrument's exits, then each funding event stamped inside the bar. What a bar draws therefore never depends on what the account holds. The stream runs on across a reset, so each episode draws afresh. `backtest`, `search` and `validate` refuse it.

<br>
<br>


## Orders

### Units

`buy(size)` and `sell(size)` take base units, or contracts on an inverse instrument. `buy_value(amount)` and `sell_value(amount)` take the quote currency, and the size such an order fills is its amount over its fill price, or over the contract's value on an inverse instrument, so it spends or raises that amount before its fee. `target(weights)` takes signed weights of account equity: `1.0` is fully long unlevered, `-0.5` half short.

### Target

At the close of bar `t`, each instrument's wanted position is the one whose notional at `close_t` is `weight * equity_t`:

```text
spot and linear     wanted = weight * equity_t / close_t                base units
inverse, V dollars  wanted = weight * equity_t * close_t / V            contracts
```

- The leg `wanted - position` is rounded toward zero to the lot. A flip becomes a reducing leg to zero and a growing leg on the other side, and every reducing leg is placed before every growing leg.
- A leg whose notional is below `tolerance * equity_t` is skipped, and so is a growing leg below the minimum notional; a reducing leg is exempt where the venue exempts reduce-only orders. A size a clamp reduces is rounded down to the lot again.
- A spot leg spends the spot wallet's cash and a perpetual leg the margin wallet's; `transfer=True` moves a spot shortfall across at the start of bar `t + 1`: what the spot legs cost at `close_t`, fees included, less what their sales raise and the spot wallet's cash.
- A negative weight on a spot instrument is refused. A second `target` in one decision replaces the first's legs.
- The difference between the weights asked for and the weights held after the fills is recorded as each instrument's drift.

**Worked example.** A flat account of 10,000 USDT targets 0.5 of BTCUSDT at a close of 60,000, with a lot of 0.001: the wanted position is `0.5 * 10,000 / 60,000 = 0.083333`, and the leg is a buy of 0.083. A wallet of 1 BTC targeting -1.0 of an inverse perpetual with 100-dollar contracts, at a close of 60,000, wants `-1.0 * 1 * 60,000 / 100 = -600` contracts. A spot wallet of 1,000 USDT and a margin wallet of 9,000, with a taker fee of 0.1%, targeting 0.5 of BTC on spot at a close of 60,000 with `transfer=True`, buy `0.5 * 10,000 / 60,000 = 0.083333` BTC for 5,000.00 and a fee of 5.00, so 4,005.00 moves to the spot wallet at the start of the next bar, before the leg fills.

### Order types

- A market order fills at the next open. A limit rests until a bar reaches it. A buy stop triggers when a bar's high reaches its trigger and a sell stop when the low does, and a triggered stop fills as a taker.
- `gtc` rests. `ioc` takes one bar and cancels the rest. `fok` fills the whole size against one bar or nothing, judged after every clamp. A market order is `ioc` unless `fok` is asked for.
- A stop that cannot fill on a bar with no volume stays armed, and a stop is consumed only by what it filled.
- An order retires when its remaining size can no longer fill ([Sizes and dust](#sizes-and-dust)).
- `reduce_only` holds on every order type and is checked when the order fills.
- `post_only` is checked at placement against the last close, where a post-only buy at or above it, or a sell at or below it, is refused, and again on the first bar: a post-only limit the first bar opens through is cancelled.
- `bracket(stop, take_profit)` places two reduce-only legs at those prices as one OCO pair. Neither has a size of its own: each closes the position it finds when it fills, so a bracket placed with its entry rests while flat and arms once the entry fills. Its legs take their side from the position they arm on and keep it, a sell stop and a sell limit behind a long, a buy stop and a buy limit behind a short, and a leg reached while the account is flat fills nothing. When either leg fills, even in part, the other is cancelled, and the filled leg's remainder keeps resting. A bracket is cancelled when the position it armed on closes or flips, so its legs never act on a position they were not placed for.
- `replace` changes a resting order's price, trigger or size and keeps its id and its OCO pair. A successful replace takes the first free slot; a refused one leaves the original in its own slot.
- `max_open_orders`, 8 by default, bounds each instrument's resting orders and its pending market orders.

**Worked example.** A sell stop at 95 fills at 95 on a bar that opens at 96 and trades down to 94, and at 93 on a bar that opens at 93, the worse of its trigger and the open. A long of 1 from 100 holds a bracket with its stop at 95 and its take-profit at 110, and the next bar opens at 100 with a low of 94 and a high of 111: both legs are reached, so under the pessimistic setting the stop fills first, at 95, and the take-profit is cancelled, and under the optimistic setting the take-profit fills at 110 and the stop is cancelled.

**Worked example of the book.** A buy limit at 95 takes the first slot and a buy limit at 94 the second. Cancelling the first frees its slot, and a `replace` that moves the 94 to 92 keeps its id and takes that first free slot, so a buy limit at 93 placed next takes the second. An `ioc` buy limit at 96 that its first bar does not reach expires after that bar, and a `fok` market buy whose size the spot wallet's cash pays for only in part fills nothing. A bracket placed in the same decision as its market entry arms when the entry fills, and a sell stop reached on a bar that printed no volume stays armed and fills on the next bar that reaches it. A bracket armed on a long of 1, with its stop at 95 and its take-profit at 110, is cancelled when a market sell of 2 flips the account short, where a buy stop at 95 and a buy limit at 110 would have closed the short at once.

### Maker, taker and the book

- A limit is a taker only on the first bar after its placement, when that bar opens at or through it, and it then fills at the taker rate. A limit that has rested through a bar is on the book and fills as a maker. Either fills at the price [Limits](#limits) gives. A `replace` that moves a limit's price is a new placement for this rule and for `post_only`.
- A limit whose price reaches the account's own resting limit of the other side on the same instrument, a buy at or above its own sell or a sell at or below its own buy, is refused, as a venue refuses an order that would trade against the same account.
- Resting orders across every instrument share one slot table and resolve in slot order, and a freed slot is taken by the next placement on any instrument.

**Worked example.** A buy limit at 99 placed at a close of 100. If the next bar opens at 98.50 it fills at 99 as a taker; if it opens at 100 and trades down to 98 it fills at 99 as a maker; and if it rests through that bar and the one after opens at 98.50, it fills at 99 as a maker. A post-only buy at 100 placed at that close is refused, and a buy at 101 placed while the account rests a sell at 100.50 is refused too.

### Barriers on an array

`backtest(weights, ..., stop=, take_profit=)` gives every position of an array run a bracket at those returns from its average entry: a long's stop at `entry * (1 - stop)` and take-profit at `entry * (1 + take_profit)`, mirrored for a short. Each is a number or a `(T, N)` array read on the row that opened the round trip, and the bracket moves when the average entry does. An instrument a barrier closed stays flat until its weight returns to zero or changes sign.

**Worked example.** A long entered at an average of 20,000 with `stop=0.04` and `take_profit=0.08` rests its stop at 19,200 and its take-profit at 21,600.

<br>
<br>


## Fills

### The fill model

Every rule is chosen in the venue. A preset uses the realistic option in each row and a hand-built venue starts from the plain one.

| Rule | Options | Plain default |
| :--- | :--- | :--- |
| limit fill | `touch`, or `through`, which needs the market to trade one tick past the limit | `touch` |
| window | the bars the volume, the volatility and the spread estimate are read over | 24 |
| liquidity | `budget(f)`, a share `f` of the trailing volume per bar and instrument, shared by every order | `budget(1.0)` |
| spread | `none`, `fixed(bps)` or `estimated` | `none` |
| slippage | `fixed(bps)` | 0 |
| impact | `none`, `linear(k)` or `sqrt(k)` | `none` |
| fees | maker and taker rates | maker 0.02%, taker 0.06% |
| sizes | lot, tick, minimum notional and maximum order quantity, or `off` | `off` |

### The window

For a fill on bar `t`, the window is the instrument's last `W` bars before `t`, or all of them when it has fewer. The trailing volume `V` is their mean volume. The volatility `sigma` is the sample standard deviation of their log returns from close to close, each measured from the bar before it, and zero when there are fewer than two.

### A taker's price

```text
price = ref * (1 + side * (s / 2 + slip + imp)), held inside [low, high]
```

`ref` is the open, or for a stop the worse of its trigger and the open; `side` is +1 for a buy and -1 for a sell; `s` is the spread as a fraction of the price; `slip` the slippage; `imp` the impact; and `low` and `high` are the bar's range, clipped where the bar is clipped. `slip` and `imp` are floored at zero, and `s / 2 + slip + imp` is held at 0.99 at most. A buy never fills above the high and a sell never below the low.

`imp = k * sigma * sqrt(q / V)` under `sqrt(k)` and `k * sigma * q / V` under `linear(k)`, where `q` is the size the liquidity budget allows the fill, before the wallet's clamps, which then apply at the price that size set.

**Worked example.** A buy of 0.141 BTC on a bar that opens at 21,012.00. An estimated spread of 0.6 bps adds 0.63, a slippage of 1 bp adds 2.10, and impact under `sqrt(0.1)`, with `sigma` 0.004 and `V` 24,800 BTC, adds `0.1 * 0.004 * sqrt(0.141 / 24,800) * 21,012 = 0.02`. The fill is at 21,014.75, below the bar's high, and the taker fee of 0.05% on 2,963.08 is 1.48.

### Limits

A limit fills at its own price, unless no trade from the last close through its bar reached that price, as when it was placed through the market or a bar that filled nothing left it there: then a buy fills at the highest of those trades and a sell at the lowest. Under `touch` a buy fills when the bar's low reaches its price; under `through` the low must be a tick below it. A sell is the mirror.

**Worked example.** A buy limit at 99.40, with a tick of 0.01, fills under `touch` on a low of 99.40, and under `through` only on a low of 99.39 or lower. A buy limit at 110 placed at a close of 100 fills as a taker on a next bar that opens at 96 with a high of 98, and since no trade from that close through the bar reached 110, it fills at 100.

### The liquidity budget

On each bar an instrument's fills may take `f * V` in all, consumed in resolution order: market orders in placement order, then resting orders by slot, buys and sells together. A forced close is not limited by it, and a bar that printed no volume fills nothing.

**Worked example.** Under `budget(0.1)` with a trailing volume of 50 BTC, the bar holds 5 BTC. Five market orders of 2 BTC fill 2, 2 and 1 BTC, and the last two fill nothing; each market order's unfilled part is cancelled, as a market order lives one bar.

### The spread estimate

`estimated` applies Ardia, Guidotti and Kroencke's estimator (Journal of Financial Economics 161, 2024) to the window. With `o`, `h`, `l` and `c` the logs of each bar's prices, `m = (h + l) / 2`, and every mean taken over the window's consecutive pairs of bars `t - 1` and `t`:

```text
r1 = m_t - o_t          r2 = o_t - m_(t-1)          r3 = m_t - c_(t-1)
r4 = c_(t-1) - m_(t-1)  r5 = o_t - c_(t-1)

tau  = 1 when h_t != l_t or l_t != c_(t-1), else 0
po   = mean(tau * [o_t != h_t]) + mean(tau * [o_t != l_t])
pc   = mean(tau * [c_(t-1) != h_(t-1)]) + mean(tau * [c_(t-1) != l_(t-1)])
pt   = mean(tau)

d1 = r1 - mean(r1) / pt * tau
d3 = r3 - mean(r3) / pt * tau
d5 = r5 - mean(r5) / pt * tau

x1 = -4 / po * d1 * r2 - 4 / pc * d3 * r4
x2 = -4 / po * d1 * r5 - 4 / pc * d5 * r4

e1 = mean(x1)    v1 = mean(x1 ** 2) - e1 ** 2
e2 = mean(x2)    v2 = mean(x2 ** 2) - e2 ** 2

s2 = (v2 * e1 + v1 * e2) / (v1 + v2), or (e1 + e2) / 2 when v1 + v2 is not positive
s  = sqrt(max(s2, 0))
```

`[a]` is 1 when `a` holds and 0 otherwise. The estimate is undefined with fewer than three bars, with fewer than two pairs where `tau` is 1, or with `po` or `pc` at zero, and an undefined estimate is a spread of zero, which the fill log records.

**Worked example.** Six bars:

| Bar | Open | High | Low | Close |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 100.0 | 100.5 | 99.8 | 100.2 |
| 2 | 100.3 | 100.6 | 100.0 | 100.0 |
| 3 | 100.1 | 100.6 | 99.9 | 100.4 |
| 4 | 100.5 | 100.8 | 100.1 | 100.1 |
| 5 | 100.2 | 100.7 | 100.0 | 100.5 |
| 6 | 100.6 | 100.9 | 100.3 | 100.3 |

give `s2 = 1.0553e-6`, a spread of 0.1027% of the price. A buy whose reference is 100.40 moves up by half of it, 0.05.

### Sizes and dust

With sizes on, an order's size is rounded down to the lot when it is placed, and so is every fill's size after its clamps, and a limit or a trigger is rounded to the nearest tick, or to Hyperliquid's five significant figures within its decimal cap. An order below the minimum notional at the last close is refused, a reducing one exempt where the venue exempts it; one above the maximum order quantity is clamped to it; and an order retires when what remains of it is less than a lot. A position that floating-point sums leave below half a lot is zeroed at the bar's close with no fee, and its value there is recorded as a dust adjustment. With sizes off, a fill of 1e-9 base units or less moves nothing, and an order retires at that remainder.

**Worked example.** With a lot of 0.001, a buy limit for 0.2 on a bar whose budget allows 0.0754 fills 0.075, rounded down to the lot, and rests on for the remaining 0.125, and a buy for 0.0004 is refused when it is placed, since it rounds to nothing.

<br>
<br>


## Accounting

### Fees

A fill of size `s` at price `p` pays `s * p * f`, at the maker or the taker rate, in the settlement currency. A round trip carries its entry fee and its exit fee exactly once: a closing fill's fee plus its share of the entry fee still standing on the position.

### Positions

A fill that crosses zero closes the old side at its average entry, realizing the PnL on that part, and opens the remainder at the fill price. A same-side add volume-weights the entry. Cash moves only with the position, and money is float64, updated in event order.

**Worked example.** A long of 1.0 at an average of 100 sells 1.5 at 110: the long closes, realizing 10.00 before fees, and a short of 0.5 opens at 110.

### Round trips

A round trip is a maximal run of one sign on one instrument: it opens when a position leaves zero and ends when it returns to zero or flips, and resizing inside it does not end it. Its net PnL is the PnL between its fill prices less its fees and the funding charged on it while open. One still open at the end is reported, marked at the last close and flagged. It reports its instrument and side; its size, the largest position it reached; its entry and exit, the average prices of its opening and its closing fills; the bars and times it opened and closed on; its fees, every fee of its fills, with a flip's fee split by size between the round trip it closes and the one it opens; its funding; its net PnL; and its return, the net PnL over its size times its entry. Its best and worst excursions count only the part of each bar the trade was in: the entry bar from the fill to the close, the exit bar from the open to the fill.

**Worked example.**

```text
bar   fill        position   round trip
10    buy 1.0      1.0       1 opens
12    buy 1.0      2.0       1
15    sell 0.5     1.5       1
20    sell 1.5     0.0       1 closes
25    sell 1.0    -1.0       2 opens
30    buy 2.0      1.0       2 closes, 3 opens
end                1.0       3 is open, marked at the last close and flagged
```

Six fills make two closed round trips and an open one, and `num_trades` is 2.

<br>
<br>


## Wallets

- The spot wallet holds cash and spot holdings. A buy is clamped to what its cash pays for, fee included, and a sell to its holding; it never borrows and is never liquidated.
- The margin wallet holds the collateral behind every perpetual, in cross margin.
- A transfer is applied at the start of the bar after it was decided, before any fill. A transfer out of the margin wallet is clamped to the cash the wallet holds, so unrealized profit cannot be moved, and so that the margin left still covers the cap's requirement; one out of the spot wallet is clamped to its cash; a clamped transfer is an order-log event. With the cap off nothing is required, so a transfer can take a margin wallet holding an unrealized loss to zero equity, at or past its trigger, and the account is then liquidated at the last marks, before the bar opens.
- Account equity is the sum of both wallets. An account whose equity is at or below zero is terminal.

**Worked example.** A spot wallet of 1,000 USDT buys 20 at an open of 100 with a taker fee of 0.1%: its cash pays for `1,000 / (100 * 1.001) = 9.99001`, so 9.99001 fills and the order log records the clamp. A margin wallet with 600 USDT of cash and 400 of unrealized profit, under a cap requirement of 200, moves at most 600 to the spot wallet, the cash it holds, though its equity less the requirement is 800. With the cap off, a margin wallet of 1,000 USDT holding a long of 10 from 100, marked at 95, moves at most 950, which leaves its equity at zero, and the long is liquidated at 95 before the next bar opens.

<br>
<br>


## Funding

### When an event is charged

- A row of the clock owns the funding events stamped after its open and up to and including its end, the open plus the interval, and they are charged at its closes after the liquidation check, on the positions still held. An account the check liquidates holds nothing at the stamp and pays nothing.
- An event stamped after one row's end and up to the next row's open, across a closure or a hole in the data, is charged at the start of the next row, before its gap segment, at the last marks, on the positions that carried through it. A debit that takes the account to its trigger there liquidates it at the start of the gap.
- An event stamped inside a row, which happens when a bar is longer than the funding interval or its boundaries miss the funding times, is charged at the row's close on the position after the bar's fills, and `intrabar` decides whether it counts toward the bar's check: under the pessimistic setting a debit counts and a credit does not. An event counted toward a check that liquidates is paid, and one not counted is not.
- A stamp within one second of a row boundary is taken as that boundary.

### The charge

At an event with rate `r` and price `p`, the margin wallet falls by `q * p * r` for each perpetual: a long pays a positive rate and a short receives it. The price is the instrument's close on the row that owns the event, or its last mark when it printed no bar there or the event fell between rows. Once the events are charged the account is checked at the closes, and if a debit has taken margin equity to its trigger, the forced close is booked at the closes, with whatever the margin wallet then holds, positive or negative, as the liquidation charge.

### Sources

| Setting | Events |
| :--- | :--- |
| `"data"` | the venue's own funding history, with each symbol's interval as it changed; a run whose data lacks the series is refused, naming the instrument and the span |
| `{"every": hours, "rate": r}` | at multiples of `hours` from 00:00 UTC |
| `{"every_bars": n, "rate": r}` | at the close of bars `n`, `2n` and so on |
| `"none"` | no events |

An `every` or `every_bars` event is charged on every perpetual held, and a `data` event on its own instrument.

**Worked example of the sources.** On hourly bars that begin at 00:00, `{"every": 8, "rate": r}` stamps events at 08:00, 16:00 and the next 00:00, each owned by the bar that ends there, and `{"every_bars": 3, "rate": r}` charges at the closes of rows 3, 6, 9 and on, counted from the first row as row 0. On daily bars, a long stopped out during a day pays nothing for that day's 08:00 event, which is charged on the position after the day's fills, and a long held through the day pays it at the day's close.

**Worked example.** Hourly bars and 8-hour funding. An event stamped 08:00:00.003 is taken as 08:00 and belongs to the bar that opened at 07:00. At a rate of 0.01%, a long of 0.5 BTC and a close of 60,000, the long pays `0.0001 * 0.5 * 60,000 = 3.00`.

**Worked example of the order.** A margin wallet of 10,000 USDT holds a long of 0.5 BTC from 60,000 at a maintenance rate of 0.5%, so its trigger is 40,201.01. The 07:00 bar falls to 40,210 and closes at 41,000, and the 08:00 event has a rate of 0.05%. The path never reaches the trigger, so the check passes; the debit of `0.0005 * 0.5 * 41,000 = 10.25` is then charged at the close, where equity is 489.75 against a requirement of 102.50, and the account survives. Counted toward the bar's low, the debit would have moved the trigger to 40,221.61 and liquidated the account on a payment it had not yet made.

Had the rate been 2%, the debit of 410.00 would leave equity of 90.00 under the requirement of 102.50, and the account would be closed at the close, 41,000, with its remaining 90.00 booked as the liquidation charge. Had the 0.05% event been stamped at 07:30, inside the bar, the pessimistic setting would count its debit toward the check, which moves the trigger to 40,221.61, above the low of 40,210, so the account would be liquidated there and the debit paid; the optimistic setting would not count it, and the account would survive as before.

**Worked example across a gap.** The same account holds its long on an instrument whose bars stop at Friday's close of 41,000 and resume at Monday's open of 40,150, with an event at a rate of 0.05% stamped on Saturday. The decision at Friday's close sees equity of 500.00, because the event has not happened yet. At the start of Monday's bar the debit of 10.25 is charged at the last mark, 41,000, which moves the trigger to 40,221.61; the gap segment from 41,000 to 40,150 crosses it, so the account is liquidated on the gap at 40,221.61, where equity and the requirement are both 100.55.

<br>
<br>


## Margin

### The cap

- Summed over the perpetuals, each position's notional over the maximum leverage of the tier its size falls in may not exceed margin equity: `sum_i N_i / L_k(i) <= E`, with equity measured before the fill's own fee, but for a flip after the fee on the side it closes, so a flip opens no more than its close leaves room for. A leverage of 0.0 removes the cap for that instrument, and a fill past the last tier's bound is clamped there.
- A growing fill is clamped to what the cap allows, and the clamp is an order-log event with the size allowed and the size asked.
- When one fill is checked, every other leg is marked at the same fraction of its own bar's move as the fill sits at on its own bar ([The path](#the-path)), or at its favourable extreme under `intrabar="optimistic"`. The fill is placed on its bar from the open to the adverse extreme of the side its instrument holds, so a flip sits where its old side leaves, and from flat of the side it opens.
- While the account is over the cap because equity fell, no fill may raise the summed requirement, and a flip must fit the cap outright. A position over the cap is never force-reduced. With margin equity at or below zero, only reducing fills pass.

**Worked example at one instrument.** A margin wallet of 10,000 USDT at 20x holds at most 200,000 of BTCUSDT notional, so a market buy of 5 BTC at an open of 60,000 fills 3.33333, and the order log records the clamp.

**Worked example.** A margin wallet of 10,000 USDT; BTCUSDT allows 20x and ETHUSDT 10x. A long of 1 BTC at 60,000 uses `60,000 / 20 = 3,000` of the cap.

- A market sell of 30 ETH at the next open, with BTC still at 60,000, has room for `(10,000 - 3,000) * 10 = 70,000` of notional, 23.333 ETH at 3,000. It fills 23.333, and the order log records the clamp.
- A sell limit on ETH that fills at its high of 3,300, on a bar where BTC falls to 54,000, sits at the end of its own adverse move, so BTC is marked at the end of its move too. Equity there is 4,000, BTC uses 2,700, and the room is 13,000 of notional, 3.94 ETH. Marked at its open instead, BTC would have left room for 21.2 ETH, which at BTC's 54,000 puts the account at 2.4 times its cap.

### Maintenance

- **Notional tiers** (Binance, Bybit, Hyperliquid): a position's requirement is `M_i = N_i * m_k - c_k` in the tier its notional falls in, and the account's is the sum. The cumulative amounts make it continuous across tier boundaries: `c_k = c_(k-1) + (m_k - m_(k-1)) * up_to_(k-1)`.
- **Quantity tiers** (OKX): the tier is chosen by the position's size in contracts, and its rate applies to the whole position.
- The loader refuses a notional table whose amounts break continuity or whose first amount is not zero, any rate not below one, any tier whose maintenance rate reaches its initial rate `1 / L_k`, and a table whose bounds do not rise, whose leverage rises or whose maintenance rate falls from one tier to the next, a leverage of 0.0 standing for no cap, since both requirements have to grow with the position.

**Worked example.** BTCUSDT's first two Binance tiers: up to 300,000 at 0.4% with no amount, then up to 800,000 at 0.5% with an amount of 300. At 300,000 of notional the first asks `0.004 * 300,000 = 1,200` and the second `0.005 * 300,000 - 300 = 1,200`.

<br>
<br>


## Liquidation

No margin wallet ends a bar owing money, a liquidation is a price on the bar rather than a check after it, and no reducing fill realizes past the point where the margin runs out. If any price a bar printed reaches the trigger, the account is liquidated on that bar: the bar's own trade prices decide.

### The path

Every perpetual held into the bar moves along one synchronous adverse path, in two segments, with `s` running from 0 to 1 on each and every leg at the same fraction of its own move:

```text
gap segment   p_i(s) = a_i + s * (o_i - a_i)    from the last mark a_i to the open o_i
bar segment   p_i(s) = o_i + s * (x_i - o_i)    from the open to the adverse extreme x_i
```

`x_i` is the low for a long and the high for a short. Equity is linear in `s` and the requirement piecewise linear, so the first crossing of any level is solved exactly, within each segment between tier breakpoints.

Legs that change during the bar join the path:
- a leg opened or grown during the bar enters at the fraction where its fill price lies on its adverse segment; a fill on the favourable side of the open enters at fraction 0, on its own segment from the fill price to the adverse extreme;
- a leg that flips is tested on both segments, the old side up to the flip's fraction on its own segment, measured from the flip's price since the wallet already holds what it realized, and the new side from there to its own adverse extreme; a flip filled past the old side's adverse extreme traded on the gap, before the bar opened, so the old side leaves at the open;
- an instrument flat at the bar's start takes the segment of the side its first fill opens;
- an instrument unavailable on the bar keeps its last mark and does not move.

Under `intrabar="optimistic"` the account is liquidated on a bar only if every path inside the bars reaches the trigger. The test is one corner per perpetual held, that instrument at its adverse extreme and every other at its favourable extreme, and the closes. If none of them reaches the trigger, a path exists that never does, from the opens to every favourable extreme, out to each adverse extreme and back in turn, and on to the closes. If one does, every path does, and the clip falls where the synchronous path meets the trigger.

### The trigger, close and clip points

- **The trigger point** is where margin equity falls to the maintenance requirement, `E = M`; with no maintenance, where equity reaches zero.
- **The close point** is where equity, less a closing fee on every position's notional, reaches zero: `E = f * sum_i N_i`, with the liquidation fee when there is no maintenance and the taker fee when there is.
- **The clip point** is where the account leaves the market: with maintenance, the trigger point; with none, the point from which the account stays at or past the close level all the way to the trigger.

For one long position in one tier:

```text
trigger, with maintenance      p = (q * e - W - c) / (q * (1 - m))
close point, no maintenance    p = (q * e - W) / (q * (1 - f))
```

For a short, `(1 - m)` becomes `(1 + m)` and `(1 - f)` becomes `(1 + f)`, with `q` negative.

### The rule

1. Any funding stamped between rows is charged at the last marks, and the path is solved for the positions carried into the bar. If the trigger is not on the path, nothing is clipped.
2. If the clip point lies on the gap segment, its start included, the account was liquidated before the bar opened: the forced close is booked first, and no perpetual order fills on the bar. The spot wallet trades on.
3. Otherwise the bar of every perpetual the account holds is clipped at the clip point's fraction of its own move, and the bar's orders resolve against the clipped bars. A perpetual the account does not hold is not clipped. Spot bars are never clipped.
4. Under the pessimistic setting, the bar's fills are first resolved in their order, each perpetual fill skipped that, when its turn comes, would reduce the position it meets at a price on the favourable side of where that position started on the bar: its open for a position carried into the bar or opened at the open, the reference of the fill that opened or flipped it for one a resting order opened. The path takes a position away from where it started, so the bar reached such a price only on a move back, which it cannot show came before the trigger. The path is then solved for the account as it stands. If it reaches the trigger the fills skipped are cancelled; otherwise the bar resolves again with them, in the same order. A sale above the price at which the bar's own buy opened the long it would close is one of them, even below the open. Under the optimistic setting they resolve in slot order with the rest.
5. **The check.** The path is solved again against the raw bars for the account after the bar's fills, every carried leg from its open and every leg opened or grown during the bar from its fill price, with any funding stamped inside the bar counted as `intrabar` decides. If the trigger is on the path, the forced close is booked at the clip point recomputed for those positions and that balance. An account already past it where its path starts closes each carried leg at its open and each new leg at its fill price.
6. Funding owned by the bar is charged at the closes, and an account a debit takes to its trigger is closed there. A margin wallet that fees have left below zero with no perpetual held, which can happen when the margin bound stops at an average entry, is set to zero, and the deficit is booked as a negative liquidation charge.

### The forced close

- **Takeover** (Binance, Bybit, OKX and every hand-built venue): every leg closes at its price at the clip point, inside its raw bar or on the gap segment. With no maintenance each close pays the liquidation fee on its notional, which at the close point leaves exactly zero. With maintenance the close pays no fee of its own. Whatever the margin wallet still holds, positive or negative, is booked as the **liquidation charge**, and the wallet ends at zero.
- **Book then backstop** (Hyperliquid): every leg is sold as a taker from its trigger price, paying spread, slippage, impact and the taker fee, held inside the raw bar and not limited by the liquidity budget, and the trader keeps what remains. Takeover applies instead if the path also reaches the backstop level, equity at the venue's stated fraction of the requirement, or if the sale's own costs would take equity below it.
- A liquidation cancels every resting perpetual order of the account. An empty margin wallet takes no growing perpetual fill. The spot wallet is untouched.

### The margin bound

1. A reducing fill is placed on its position's path at its fraction. A fill on the favourable side of its open, or on a bar with no adverse move, is at fraction 0, and so is a fill past the adverse extreme, which traded on the gap before the bar opened.
2. The account's close-level equity is evaluated with every other leg at the same fraction, or at its favourable extreme under `intrabar="optimistic"`, and with the fill's own fee in place of the closing fee on the size it closes.
3. If that equity is below zero, the fill is lifted toward the price where it is zero, or a purchase toward zero where no such price lies above it, never outside its instrument's clipped bar nor past its position's average entry. The lift the bar holds moves the fill's price and is recorded as its adjustment; the part it cannot hold, times the size closed, is paid into the margin wallet, net of the fee it would have added to a sale or with the fee it would have saved a purchase. Both are booked as a negative liquidation charge, so the price PnL stays measured from the fill's reference.

**Worked example.** A margin wallet of 50.5, a long of 10 at 100, a fee of 0.1%, and a market sell on a bar that trades at 95 throughout. The close point is where `50.5 + 10 * (p - 100) = 0.001 * 10 * p`, so `p = 949.5 / 9.99 = 95.045045`. The sale fills at 95, the bar's high, and `(95.045045 - 95) * 10 * 0.999 = 0.45` is booked as a negative liquidation charge.

### The error and its direction

For linear contracts, margin equity less the requirement rises with a long leg's price and falls with a short leg's, at a fixed quantity. Every real path stays inside the box the bars define, and the worst point of that box is the corner where every leg sits at its adverse extreme at once, where the synchronous path ends. So the pessimistic setting can only err toward liquidating, and the optimistic setting only the other way. The exceptions, each named in a result's assumptions with its direction:
- an instrument unavailable on a bar is frozen at its last mark, so a liquidation on a hidden path is not seen (optimistic);
- liquidation in slices is not modelled (pessimistic for large positions);
- auto-deleveraging is not modelled (optimistic for winners);
- a flip that rule 4 holds back is held whole, the side it would open with it, so an exit of that side, which the bar may have reached only after the trigger, can fill (optimistic).

### Worked for one instrument

A margin wallet of 10,000 USDT holds a long of 0.5 BTC from 60,000; the taker and liquidation fee is 0.05% and the maker fee 0.02%. Equity at price `p` is `10,000 + 0.5 * (p - 60,000) = 0.5p - 20,000`.

- **No maintenance; a bar that opens at 41,500 with a low of 39,000.**
  - The trigger is where equity is zero, 40,000.00. The close point is where `0.5p - 20,000 = 0.00025p`, 40,020.01, and that is the clip point.
  - With nothing resting, the check closes the position at 40,020.01: equity there is 10.005, the fee is 10.005, the wallet ends at zero and the charge is zero.
  - With a sell stop resting at 41,000, the stop fills at 41,000 on the clipped bar, the check finds the account flat, and it keeps `10,000 - 9,500 - 10.25 = 489.75`.
  - With a sell limit resting at 42,000 instead, on a bar whose high is 42,500, the limit lies on the favourable side of the open and the path still reaches the trigger, so under the pessimistic setting rule 4 cancels it and the account is liquidated at 40,020.01. Under the optimistic setting the limit fills first and the account keeps `10,000 - 9,000 - 4.20 = 995.80`.
  - If the last mark was 40,500 and the bar opens at 40,010, the gap crosses the clip point before the open, so the account was liquidated before the bar opened, at 40,020.01, and nothing on the bar fills.
- **A maintenance rate of 0.5%; a bar with a low of 40,150.**
  - The requirement is `0.0025p`, so the trigger is where `0.5p - 20,000 = 0.0025p`: 40,201.01, where equity and the requirement are both 100.50. That is the clip point.
  - Under takeover the position closes at 40,201.01, and the 100.50 the wallet holds is the liquidation charge.
  - Under book then backstop the position is sold at 40,201.01 as a taker and pays the taker fee of 10.05, so on a venue that takes no clearance fee the trader keeps 90.45 less the sale's spread, slippage and impact. The backstop level, two thirds of the requirement, is where `0.5p - 20,000 = 0.0016667p`, 40,133.78, which this bar's low of 40,150 does not reach. A sale costing more than 33.50 would take equity below that level at the trigger, and takeover would apply.

### Worked for two instruments

A margin wallet of 10,000 USDT holds a long of 1 BTC and a short of 20 ETH, entered and marked at 60,000 and 3,000, with no maintenance and a fee of 0.05%. The cap holds: `60,000 / 20 + 60,000 / 10 = 9,000`. On the next bar BTC opens at 60,000 with a low of 54,000, and ETH opens at 3,000 with a high of 3,300, so the gap segment is empty.

- Along the path, at fraction `s`, BTC is at `60,000 - 6,000s` and ETH at `3,000 + 300s`, so equity is `10,000 - 12,000s`. The total notional stays 120,000, so the closing fee is a constant 60.
- The trigger: `10,000 - 12,000s = 0` gives `s = 0.8333`.
- The close point: `10,000 - 12,000s = 60` gives `s = 0.8283`, where BTC is at 55,030.00 and ETH at 3,248.50. With no maintenance this is the clip point.
- Both bars are clipped there and both legs close at those prices; equity there is 60, the fee is 60, and the wallet ends at zero.

Under the optimistic setting, with BTC's high and ETH's low at their opens, the corners are BTC at 54,000 with ETH at 3,000, and ETH at 3,300 with BTC at 60,000, and the bars close at 57,000 and 3,150. Equity is 4,000 at each corner and at the closes, so the account survives the bar. Had the bars closed at 55,000 and 3,250, equity at the closes would be zero: every path ends there, so the optimistic setting liquidates too, at the synchronous path's clip.

<br>
<br>


## Inverse contracts

An inverse perpetual is quoted in dollars and settled in its coin. A contract is worth a fixed `V` dollars, so a position of `n` contracts, signed, holds `u = n * V` dollars, and everything the account counts is in the coin:

| Quantity | Linear | Inverse |
| :--- | :--- | :--- |
| profit from `e` to `p` | `q * (p - e)` | `u * (1 / e - 1 / p)` |
| notional | `abs(q) * p` | `abs(u) / p` |
| a fee at rate `f` | `abs(q) * p * f` | `abs(u) / p * f` |
| a funding charge at rate `r` | `q * p * r` | `u / p * r` |
| the average entry of fills `k` | `sum(q_k * p_k) / sum(q_k)` | `sum(n_k) / sum(n_k / p_k)` |
| the size a value order of `A` fills | `A / p` | `A / V` contracts |

- Every inverse quantity is linear in `1 / p`, so the tiers, the three points and the margin bound are solved as for linear contracts, in the reciprocal of the price, and each crossing is read back as a price. The tiers are by notional in the coin.
- The average entry is the harmonic mean of the fill prices, weighted by contracts, which keeps a position's profit `u * (1 / e - 1 / p)` equal to the sum of its fills' profits.
- What a fill paid beyond its reference is counted in the coin, `abs(u * (1 / ref - 1 / price))`, and its spread, slippage and impact share it in proportion to their parts of the price.
- An inverse account holds one perpetual, its margin wallet holds the coin, it has no spot wallet to transfer to, and its path has one leg.
- Results are in the coin. The ledger also values equity in dollars at each close, as `equity_in_quote`.
- For one position in one tier, the trigger is `p = u * (1 + m) / (W + u / e + c)` for a long and `p = u * (1 - m) / (W + u / e + c)` for a short; with no maintenance the close point takes `f` in place of `m` and drops `c`.

**Worked example.** Contracts of 100 dollars and a maintenance rate of 0.5%.
- A wallet of 1 BTC with a short of 600 contracts from 60,000, which is 60,000 dollars, one coin's worth. Its equity is `1 - 60,000 * (1 / 60,000 - 1 / p) = 60,000 / p` coins, worth 60,000 dollars at every price, and its requirement, `300 / p`, is always below it, so it is never liquidated. At a rate of 0.01% and a price of 60,000 each funding event pays it 0.0001 BTC.
- A wallet of 0.5 BTC with a long of 600 contracts from 60,000 has equity `1.5 - 60,000 / p` and requirement `300 / p`, so its trigger is `60,000 * 1.005 / 1.5 = 40,200`, where both are 0.0074627 BTC.
- A buy of 300 contracts at 60,000 and another at 40,000 make a long of 600 with an average entry of `600 / (300 / 60,000 + 300 / 40,000) = 48,000`. At 50,000 its profit is `60,000 * (1 / 48,000 - 1 / 50,000) = 0.05` BTC, the sum of the fills' own, `-0.10 + 0.15`. A `buy_value(6,000)` buys 60 contracts at any price, and a taker fee of 0.05% on 600 contracts filled at 60,000 is `60,000 / 60,000 * 0.0005 = 0.0005` BTC.

<br>
<br>


## What a run records

### The ledger

One row per bar from the first, where row 0 is the opening balance, so no curve is one shorter than the data.
- Per account: equity, each wallet's cash, gross and net exposure, leverage, margin used, the maintenance requirement, and that bar's fees, funding, spread, slippage, impact, liquidation charge and dust adjustment.
- Per instrument: position, average entry, mark, notional, unrealized PnL, weight and drift.

### The ledger identity

Each bar, the change in equity equals the price PnL less spread, slippage, impact, fees, funding, the liquidation charge and the dust adjustment. Price PnL is measured from reference prices, a fill's reference and then each bar's mark, and a fill's spread, slippage and impact are what it paid beyond its reference, each in proportion when the bar's range held its price back. Every term is computed on its own, so none is a residual. The identity holds within an error bound derived from the magnitudes involved. On a run that ends flat with no liquidation charge and no dust adjustment, the round trips' net PnL sums to the change in equity.

**Worked example.** A bar carries 0.5 BTC from a mark of 60,000 and buys 0.1 BTC. The reference is the open, 60,300. The buy pays an estimated half-spread of 3.00, a slippage of 1 bp (6.03) and the taker fee of 0.05%, for a fill at 60,309.03. The bar closes at 60,500.
- Price PnL: `0.5 * (60,500 - 60,000) + 0.1 * (60,500 - 60,300) = 250 + 20 = 270.00`.
- Costs: spread `0.1 * 3.00 = 0.30`, slippage `0.1 * 6.03 = 0.603`, fee `0.0005 * 0.1 * 60,309.03 = 3.0155`.
- Change in equity: `270.00 - 0.30 - 0.603 - 3.0155 = 266.0815`. From the fill directly, `250 + 0.1 * (60,500 - 60,309.03) - 3.0155` gives the same 266.0815.

### The fill log and the order log

- The fill log has a row per fill: time, instrument, order id, side, size and price; the reference price; spread, slippage, impact and fee; any adjustment that held the price inside the bar or applied the margin bound; maker or taker; and the reason, one of `market`, `limit`, `stop`, `take_profit`, `rebalance`, `seam`, `liquidation` and `delist`.
- The order log has every order's events, each with its reason: `placed`, `armed`, `clamped`, `refused`, `not_filled`, `expired`, `cancelled` and `filled`.

### Recording levels

| Level | Keeps |
| :--- | :--- |
| `"full"` | everything |
| `"account"` | the account's columns and one row per round trip |
| `"transitions"` | what offline reinforcement learning needs |
| `"none"` | the final state only; `backtest` refuses it |

<br>
<br>


## Numbers that are the same everywhere

- Basic arithmetic and `sqrt` are correctly rounded everywhere, but `ln`, `exp` and `powf` may differ by platform, so integer powers are written as products and every other transcendental function goes through the `libm` crate.
- Random draws come from named streams, PCG64 in Python and ChaCha in Rust, each derived from the run's seed through a NumPy `SeedSequence` and recorded with it.
- Sums run in a fixed instrument order, the batch has no cross-environment reductions, and the statistics a search ranks on are summed in a fixed order.
- Compensated sums are used only for totals reported over many bars, such as fees, funding and turnover; cash and positions stay plain arithmetic in event order.

<br>
<br>


## The low-level binding

Below the Python surface, one binding drives the engine directly: it takes the candles, a venue record and a per-bar action script, and returns the ledger, the fill log and the order log. A cancel or a replacement names its order by a reference resolved at run time, an earlier placement in the script or the oldest order still live, so a refused placement leaves the script meaningful. The reference simulator and the tests drive the engine through it.
