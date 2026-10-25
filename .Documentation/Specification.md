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

This page is every rule the engine applies to money: when an order fills and at what price, what each fill costs, when funding is charged, how margin is measured, where an account is liquidated, what a run records, how a search scores and checks its trials, how a validation divides the bars and runs its paths, and how far a run's statistics can be trusted. Each rule is stated as an equation where it has one and followed by a worked example, and every worked example is a test, so a number on this page is a number the engine produces. Why each rule is the way it is lives in [Decisions](Decisions.md).

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
- An instrument enters at its first bar. At its delisting time, on the first row that opens at or after it and before that row's orders fill, its waiting orders are cancelled, those waiting for their open in the order they were placed and then the resting ones by slot, and its position is closed at its last close with no fee, each with the reason `delist`; from then on an order on it is refused, and an account that opens on that row or later opens with it delisted. A delisting time at or before the open of its instrument's last bar is refused, since that bar would trade after it.

**Worked example.** A stock's daily bars on Thursday, Friday and Monday are three consecutive rows. A 3-bar average on Monday averages Thursday, Friday and Monday; a market order decided at Friday's close fills at Monday's open; and a stop that Monday's open gaps through fills at that open.

**Worked example of a delisting.** An instrument delisted at 12:00, whose last hourly bar opened at 11:00 and closed at 50.00, while another instrument goes on printing. A long of 2 held into 12:00 closes on the 12:00 row at 50.00 with no fee and the reason `delist`, and a market order still waiting on it is cancelled.

### Annualisation

`periods_per_year = (T - 1) * 365 / D`, where `T` is the number of rows on the clock and `D` the days from the first row's open to the last row's. It is taken once over the whole Data, so every slice of it annualises alike, and a Data needs two rows to have one. A search's in-sample bars are a Data of their own, which annualises over its own rows ([Search](#search)), and its held-out run over the whole Data's. A validation's pieces, fits, windows and paths annualise as its in-sample bars do ([Fits](#fits)).

**Worked example.** Hourly bars from 00:00 on 1 January of a year that is not a leap year to 00:00 on 1 January of the next are 8,761 rows over 365 days: `8,760 * 365 / 365 = 8,760`. A stock's 253 daily bars from one 2 January to the next annualise at 252.

### Coarser bars and outside series

- `resample(interval)` needs a whole multiple of the Data's interval. A coarse bar takes the first open, the highest high, the lowest low and the last close of its bars and the sum of their volumes, and is available if any of them was. Coarse bars align to 00:00 UTC and weeks to Monday, moved by the offset of the Data's own steps from that grid, so that every bar lies inside one coarse bar: hourly bars that open at half past the hour give coarse bars that do too, and 4-hour bars that open at 02:00 give days that start at 02:00. A coarse bar the data does not cover to its end is dropped, and the first holds the bars from the data's first, wherever in it the data begins.
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
4. An instrument whose delisting time the row reached is delisted ([The clock](#the-clock)), its position closed at its last close where the path left it one. Pending market orders then fill at the open of the clipped bars, in the order they were placed. Where they opened, grew or flipped a perpetual of an account with maintenance, the bars are clipped again for the resting orders ([The rule](#the-rule), 3), as they are after each such fill of a resting order for the orders after it.
5. Resting limits and stops resolve against the clipped bars in slot order. The exits of the position each fill meets, as the fills before it left that position, are reordered by the PnL per unit each would realize at its fill price for all it asks, a stop's costs included, worst first under the pessimistic setting, among the slots those exits hold, so a position the bar's own fills open has its exits ordered as a carried one's are; exits on different instruments are never reordered. An order that can only reduce meets a position a fill inside the bar opened, flipped or grew on the part of the bar after the last such fill ([What a bar cannot show](#what-a-bar-cannot-show)). Every reducing fill passes the margin bound.
6. Under the pessimistic setting, the perpetual fills that would reduce the position they meet, with their reference on the favourable side of where that position started on the bar, are held back when the path without them still reaches the trigger.
7. The liquidation check runs along the path for the account as it now stands, and a liquidation cancels every perpetual order still standing. An ioc or fok limit the check leaves standing then expires, its one bar over.
8. Funding owned by the bar is charged at the closes on the positions still held.
9. Every position is marked at its close, a position below half a lot is swept as dust where sizes hold ([Sizes and dust](#sizes-and-dust)), and the ledger row is written.

An order decided at the close of bar `t` never fills on bar `t`.

### What a bar cannot show

A bar gives four prices and no order between them. The venue's `intrabar` setting decides every event that order would settle, and `"pessimistic"` is the default for every venue and every preset.

| Event | `"pessimistic"` | `"optimistic"` |
| :--- | :--- | :--- |
| a stop and a take-profit of one position both reached | the worse exit fills first | the better exit fills first |
| a take-profit and the liquidation trigger both reached | the take-profit is held back | the take-profit fills first |
| a fill inside the bar that opens, flips or grows a position, and the bar's extreme on the open's side of it, where that extreme favours the position | it came before the fill | it came after the fill |
| a funding event stamped inside the bar | a debit counts toward the liquidation and a credit does not | a credit counts toward it and a debit does not |
| several instruments, each reaching its worst price | they reach them at the same moment | each reaches it while the others sit at their best |
| a liquidation's sale on the book and the backstop level both reached | the backstop takes over | the sale finishes first |

A fill inside the bar is a resting order's fill at a price the open had not reached: a limit beyond the open, or a stop whose trigger lies beyond it. The price came to it from the open, so the bar's extreme beyond the fill came after it, while the one on the open's side may have come on either side of it. That extreme favours a position a limit opened or grew, and the setting places it; it works against one a stop opened or grew, and came after the fill under either setting, as the check runs that position from its fill price to its adverse extreme ([The path](#the-path)). An order that can only reduce, a reduce-only order or a sale on spot, meets a position such a fill last opened, flipped or grew, where it would reduce it, on the part of the bar after that fill, since what the fill added traded only after it and one order cannot tell those units from the rest, read as a bar of its own: it opens at the fill's reference, or at the clip where one after the fill tightened past it ([The rule](#the-rule), 3), and closes at the bar's close, reaches the extreme beyond the fill, and reaches the one on the open's side where that extreme came after the fill, or otherwise no further that way than the fill's reference and the close. [Fills](#fills) read that part as the bar, with the fill's reference as its last close, so a stop it reaches fills at the worse of its trigger and that reference. A fill at the open, a market order's or a resting order's the open had reached, leaves the whole bar after it.

**Worked example.** A buy limit at 95 placed while flat, with a bracket at 90 and 97 placed after it, meets a bar that opens at 100, reaches 100.50 and 94, and closes at 95.50. The entry fills at 95, inside the bar. Under the pessimistic setting the high came before the fill, so the part after it runs from 95 down to 94 and up to the close: the take-profit rests and the long is held. Under the optimistic setting the high came after it, and the take-profit fills at 97 as a maker. A buy stop at 105 placed while flat, with a bracket at 102 and 110, meets a bar that opens at 100, reaches 108 and 98, and closes at 103. The entry fills at 105, and both extremes came after it: the high, which the price had not reached, and the low, which works against the long. Under either setting the stop leg fills at 102, where the price crossed it after the fill, rather than at the open.

`"random"` draws one of the two for each event from the stream named `intrabar` in the run's seed, which in a batch is named with the environment's index as well. Every bar draws in one order whether or not each event arises: the instruments together, the exits held back, each instrument's exits, each instrument's fill inside the bar, each funding event the row owns, then the backstop. What a bar draws therefore never depends on what the account holds. The stream runs on across a reset, so each episode draws afresh. `backtest`, `search`, `validate` and `labels` refuse it.

<br>
<br>


## Orders

### Units

`buy(size)` and `sell(size)` take base units, or contracts on an inverse instrument. `buy_value(amount)` and `sell_value(amount)` take the quote currency, and the size such an order fills is its amount over its fill price, or over the contract's value on an inverse instrument, so it spends or raises that amount before its fee. `target(weights)` takes signed weights of account equity: `1.0` is fully long unlevered, `-0.5` half short.

A value order is a market order. Its amount over the open sizes the taker's fill that prices it, and the amount over that price is the size it fills, up to the bar's budget; where the venue's sizes are on, each size is held to the largest order and rounded down to the lot, and an amount below the minimum notional is refused as it is placed, a reduce-only one spared where the venue spares it. On a bar whose budget is spent, no fill prices a value order, and it asks its amount over the open, held to the largest order and rounded down to the lot. On an inverse contract the amount, in dollars, buys the contracts it is worth, a size the order has as it is placed, so it is a market order of that size from its placement on.

**Worked example.** `buy_value(1,000)` on BTCUSDT, with 1% slippage and a next open of 20,000, fills `1,000 / 20,200 = 0.049505` BTC at 20,200, and with a lot of 0.001, 0.049. On a contract worth 100 dollars, `buy_value(1,000)` buys 10 contracts at any price.

### Target

At the close of bar `t`, each instrument's wanted position is the one whose notional at `close_t` is `weight * equity_t`:

```text
spot and linear     wanted = weight * equity_t / close_t                base units
inverse, V dollars  wanted = weight * equity_t * close_t / V            contracts
```

- The leg `wanted - position` is rounded toward zero to the lot. A flip becomes a reducing leg to zero and a growing leg on the other side, and every reducing leg is placed before every growing leg.
- A leg whose notional is below `tolerance * equity_t` is skipped, judged before a flip splits it. A part below the minimum notional is skipped too, a reducing part only where the venue does not exempt reduce-only orders, and so is a leg or a part of `1e-9` or less, which no fill could book. A size a clamp reduces is rounded down to the lot again.
- An instrument yet to print its first bar has no close to size a leg on: a weight of zero there asks for nothing, and any other places a leg that is refused, as any order there is.
- A spot leg spends the spot wallet's cash and a perpetual leg the margin wallet's; `transfer=True` moves a spot shortfall across at the start of bar `t + 1`: what the spot legs it placed cost at `close_t`, fees included, less what their sales raise and the spot wallet's cash.
- A negative weight on a spot instrument is refused. A second `target` in one decision replaces the first: its legs, the shortfall it moved and the weights it asked for. A target in a later decision replaces the legs earlier targets left waiting for the next bar of an instrument it names, and sizes its own from the position as it stands; a leg it replaced stays replaced, even when a second target in its decision replaces it in turn.
- From the row after a target is decided, each instrument it named records its drift, the weight asked for less the weight the row holds at its close, until a later target names it again or a validation's seam forgets every target ([Paths and seams](#paths-and-seams)). An instrument no target has named records none, a NaN in the ledger.

**Worked example.** A flat account of 10,000 USDT targets 0.5 of BTCUSDT at a close of 60,000, with a lot of 0.001: the wanted position is `0.5 * 10,000 / 60,000 = 0.083333`, and the leg is a buy of 0.083. Without a lot, and with no fees, the leg buys 0.083333 at a next open of 60,000, and a close of 66,000 leaves equity of 10,500 holding 5,500 of bitcoin, a weight of 0.5238, so that row's drift is `0.5 - 0.5238 = -0.0238`. A wallet of 1 BTC targeting -1.0 of an inverse perpetual with 100-dollar contracts, at a close of 60,000, wants `-1.0 * 1 * 60,000 / 100 = -600` contracts. A spot wallet of 1,000 USDT and a margin wallet of 9,000, with a taker fee of 0.1%, targeting 0.5 of BTC on spot at a close of 60,000 with `transfer=True`, buy `0.5 * 10,000 / 60,000 = 0.083333` BTC for 5,000.00 and a fee of 5.00, so 4,005.00 moves to the spot wallet at the start of the next bar, before the leg fills.

### Order types

- A market order fills at the next open. A limit rests until a bar reaches it. A buy stop triggers when a bar's high reaches its trigger and a sell stop when the low does, and a triggered stop fills as a taker.
- `gtc` rests. `ioc` takes one bar and cancels the rest. `fok` fills the whole size against one bar or nothing, judged after every clamp. A market order is `ioc` unless `fok` is asked for.
- A stop that cannot fill on a bar with no volume stays armed, and a stop is consumed only by what it filled.
- An order retires when its remaining size can no longer fill ([Sizes and dust](#sizes-and-dust)).
- `reduce_only` holds on every order type and is checked when the order fills.
- `post_only` is checked at placement against the last close, where a post-only buy at or above it, or a sell at or below it, is refused, and again on the first bar: a post-only limit the first bar opens at or through is cancelled at the open, and one that meets the part of that bar after a fill ([What a bar cannot show](#what-a-bar-cannot-show)) opening at or through it is cancelled there, so a post-only limit never fills as a taker.
- `bracket(stop, take_profit)` places two reduce-only legs at those prices as one OCO pair. Neither has a size of its own: each closes the position it finds when it fills. A bracket placed while its instrument's position is held arms at once; one placed while flat rests until the first fill that opens a position there, a market order's or a resting order's, and arms on it, so a leg in a later slot than that fill can fill on the same bar. Its legs take their side from the position they arm on and keep it, a sell stop and a sell limit behind a long, a buy stop and a buy limit behind a short, and a leg reached while the account is flat fills nothing. When either leg fills, even in part, the other is cancelled, and the filled leg's remainder keeps resting while that position stays open. A bracket is cancelled when the position it armed on closes or flips, the leg whose fill closed it included, so its legs never act on a position they were not placed for.
- `replace` changes a resting order's price, trigger or size and keeps its id and its OCO pair. A successful replace takes the first free slot; a refused one leaves the original in its own slot.
- `cancel` and `cancel_all` reach every order not yet filled: a resting one, and a market or value order waiting for its open, a target's leg among them, whose target keeps the weights it asked for. `cancel_all` takes the waiting orders in the order they were placed and then the resting ones by slot, as a validation's seam does ([Paths and seams](#paths-and-seams)).
- `max_open_orders`, 8 by default, bounds each instrument's resting orders and its pending market orders.

**Worked example.** A sell stop at 95 fills at 95 on a bar that opens at 96 and trades down to 94, and at 93 on a bar that opens at 93, the worse of its trigger and the open. A long of 1 from 100 holds a bracket with its stop at 95 and its take-profit at 110, and the next bar opens at 100 with a low of 94 and a high of 111: both legs are reached, so under the pessimistic setting the stop fills first, at 95, and the take-profit is cancelled, and under the optimistic setting the take-profit fills at 110 and the stop is cancelled.

**Worked example of the book.** A buy limit at 95 takes the first slot and a buy limit at 94 the second. Cancelling the first frees its slot, and a `replace` that moves the 94 to 92 keeps its id and takes that first free slot, so a buy limit at 93 placed next takes the second. An `ioc` buy limit at 96 that its first bar does not reach expires after that bar, and a `fok` market buy whose size the spot wallet's cash pays for only in part fills nothing. A bracket placed in the same decision as its market entry arms when the entry fills, and a sell stop reached on a bar that printed no volume stays armed and fills on the next bar that reaches it. A buy limit at 95 placed while flat, with a bracket at 90 and 110 placed after it, meets a bar that opens at 100 and trades down to 85: the entry fills at 95, the bracket arms on that long, and its stop closes it at 90 on the same bar. A bracket armed on a long of 1, with its stop at 95 and its take-profit at 110, is cancelled when a market sell of 2 flips the account short, where a buy stop at 95 and a buy limit at 110 would have closed the short at once.

### Maker, taker and the book

- A limit is a taker only on the first bar after its placement, when that bar opens at or through it, and it then fills at the taker rate. A limit that has rested through a bar is on the book and fills as a maker. Either fills at the price [Limits](#limits) gives. A `replace` that moves a limit's price is a new placement for this rule and for `post_only`, and a bracket leg's arming is one for this rule, its first bar the one it arms on.
- A limit whose price reaches the account's own resting limit of the other side on the same instrument, a buy at or above its own sell or a sell at or below its own buy, is refused, as a venue refuses an order that would trade against the same account. A bracket's take-profit is held to this when it takes its side: at placement when its instrument's position is held, and otherwise as it arms, when one that would trade against the account's own is cancelled. A leg yet to arm has no side, and nothing is refused against it.
- Resting orders across every instrument share one slot table and resolve in slot order, and a freed slot is taken by the next placement on any instrument.

**Worked example.** A buy limit at 99 placed at a close of 100. If the next bar opens at 98.50 it fills at 99 as a taker; if it opens at 100 and trades down to 98 it fills at 99 as a maker; and if it rests through that bar and the one after opens at 98.50, it fills at 99 as a maker. A post-only buy at 100 placed at that close is refused, and a buy at 101 placed while the account rests a sell at 100.50 is refused too.

### An array

`backtest(array, ...)` reads row `t` of a `(T, N)` array as the decision at the close of bar `t`, in the unit `sizing` names.
- `sizing="weight"`: each value is a weight of equity, reached as [Target](#target) reaches it, with the run's `tolerance`.
- `sizing="quantity"`: each value is the position wanted, in base units or contracts, and the leg is `wanted - position`, rounded, split and skipped as a target's is. The weight it records as asked, for its drift, is the wanted position's notional at each row's close over that row's equity.
- A NaN holds: the instrument goes unnamed on that row, so its position, any leg it has waiting and its drift stand.
- The last row is never read, since no bar follows it to fill on. A row on which none of the instruments printed is one the account never sees: its values are taken at the last close before it, a later row's replacing an earlier one's, and a row before the account's first is not read.
- A negative value on a spot instrument is refused before the run.

### Barriers on an array

`backtest(array, ..., stop=, take_profit=)` gives every round trip of an array run a bracket at those returns from its average entry: a long's stop at `entry * (1 - stop)` and take-profit at `entry * (1 + take_profit)`, a short's at `entry * (1 + stop)` and `entry * (1 - take_profit)`, on an inverse contract's quoted price.
- Each is a number or a `(T, N)` array read on the row whose decision opened the round trip, where a NaN leaves that side without a barrier. A price at or below zero, a long's stop or a short's take-profit of 1 or more, is not placed.
- The bracket is placed at the close of the bar the round trip opened on, from flat or by a flip, once its average entry is known, so the bar its entry filled on runs without it. At each later close after a fill moved the average entry, its legs still resting are replaced at the new entry, with the returns of the row that opened the round trip. A round trip keeps the bracket it opened with, so a leg its partner's fill cancelled is not placed again.
- From the bar a barrier fills on, even in part, the instrument places nothing more on that side: a row asking for that side leaves it unnamed, and the leg's remainder closes the rest. A row asking for zero or for the other side names it again, and a NaN holds this as it holds the rest.
- At each close the brackets are placed and moved instrument by instrument, a stop before its take-profit, and before the row's target is.

**Worked example.** A long entered at an average of 20,000 with `stop=0.04` and `take_profit=0.08` rests its stop at 19,200 and its take-profit at 21,600 from the close of the bar its entry filled on. An add of the same size that fills at 21,000 takes the average entry to 20,500 and moves them to 19,680 and 22,140 at the close of its bar, and a short entered at 20,000 rests its stop at 20,800 and its take-profit at 18,400. With a weight of 1.0 on every row, a stop that fills leaves the instrument flat until a row asks for zero or less.

### A strategy

A Strategy's `init(data)` runs once over the whole Data before the first bar, and `next(bar, account)` runs at the close of every bar from `warmup` on, `warmup` counting the Data's rows, except the last, which no bar follows to fill on, and a row none of the instruments printed, which the account never sees. What `next` places at a close fills on a later bar, never on that one. The account it reads is the one that close left, with the events since the last close, and `bar.history(n)` holds the last `n` bars up to and including that one. A run of a search's causality check calls `next` at the close of its last row too, where that row is printed and at or after the warm-up, so that the decision there can be compared, and nothing placed there fills ([The causality check](#the-causality-check)). At a validation's seam the incoming window's `on_inherit(bar, account)` is called before its first `next`, its events and its record starting there ([Paths and seams](#paths-and-seams)).

<br>
<br>


## Fills

### The fill model

Every rule is chosen in the venue. A preset uses the realistic option in each row and a hand-built venue starts from the plain one.

| Rule | Options | Plain default |
| :--- | :--- | :--- |
| limit fill | `touch`, or `through`, which needs the market to trade half a step of the instrument's price rule past the limit, whether or not sizes hold | `touch` |
| window | the bars the volume, the volatility and the spread estimate are read over | 24 |
| liquidity | `budget(f)`, a share `f` of the trailing volume per bar and instrument, shared by every order | `budget(1.0)` |
| spread | `none`, `fixed(bps)` or `estimated` | `none` |
| slippage | `fixed(bps)` | 0 |
| impact | `none`, `linear(k)` or `sqrt(k)` | `none` |
| fees | maker and taker rates, which an instrument may set apart from the venue's | maker 0.02%, taker 0.06% |
| sizes | rounding to the lot and the price rule, the minimum notional and the maximum order quantity, or `off` | `off` |

### The window

For a fill on bar `t`, the window is the instrument's last `W` bars before `t`, or all of them when it has fewer. The trailing volume `V` is their mean volume, in the units an order's size is in: base units, or contracts on an inverse instrument. The volatility `sigma` is the sample standard deviation of their log returns from close to close, each measured from the bar before it, and zero when there are fewer than two.

### A taker's price

```text
price = ref * (1 + side * (s / 2 + slip + imp)), held inside [low, high]
```

`ref` is the open, or for a stop the worse of its trigger and the open; `side` is +1 for a buy and -1 for a sell; `s` is the spread as a fraction of the price; `slip` the slippage; `imp` the impact; and `low` and `high` are the bar's range, clipped where the bar is clipped. `slip` and `imp` are floored at zero, and `s / 2 + slip + imp` is held at 0.99 at most. A buy never fills above the high and a sell never below the low. On an inverse contract the rates move the reciprocal of the price, `1 / price = (1 / ref) * (1 - side * (s / 2 + slip + imp))`, so each costs its rate of the coin notional at the reference, as a fee costs its rate of the notional at the fill ([Inverse contracts](#inverse-contracts)).

`imp = k * sigma * sqrt(q / V)` under `sqrt(k)` and `k * sigma * q / V` under `linear(k)`, where `q` is the size the liquidity budget allows the fill, before the wallet's clamps, which then apply at the price that size set; an order given in the quote currency is sized for this at its reference. With `V` at zero the budget fills nothing, so only a sale on the book ([The forced close](#the-forced-close)) can fill there, and it pays no impact.

**Worked example.** A buy of 0.141 BTC on a bar that opens at 21,012.00. An estimated spread of 0.6 bps adds 0.63, a slippage of 1 bp adds 2.10, and impact under `sqrt(0.1)`, with `sigma` 0.004 and `V` 24,800 BTC, adds `0.1 * 0.004 * sqrt(0.141 / 24,800) * 21,012 = 0.02`. The fill is at 21,014.75, below the bar's high, and the taker fee of 0.05% on 2,963.08 is 1.48.

### Limits

A limit fills at its own price, unless no trade from the last close through its bar reached that price, as when it was placed through the market or a bar that filled nothing left it there: then a buy fills at the highest of those trades and a sell at the lowest. Under `touch` a buy fills when the bar's low reaches its price; under `through` the low must reach half a step of the instrument's price rule below it, the step taken at the limit's price whether or not sizes hold, and with no price rule `through` is `touch` ([Sizes and dust](#sizes-and-dust)). A sell is the mirror, and on an inverse contract the step and the prices are in dollars.

**Worked example.** A buy limit at 99.40, with a tick of 0.01, fills under `touch` on a low of 99.40, and under `through` only on a low of 99.395 or lower, 99.39 where prices sit on the tick. A buy limit at 110 placed at a close of 100 fills as a taker on a next bar that opens at 96 with a high of 98, and since no trade from that close through the bar reached 110, it fills at 100.

### The liquidity budget

On each bar an instrument's fills may take `f * V` in all, consumed in resolution order: market orders in placement order, then resting orders in the order they resolve ([The order of events](#the-order-of-events), 5), buys and sells together. A forced close is not limited by it, even on a bar that printed no volume, where every other order fills nothing.

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

`[a]` is 1 when `a` holds and 0 otherwise. The estimate is undefined with fewer than three bars, with fewer than two pairs where `tau` is 1, or with `po` or `pc` at zero, and an undefined estimate is a spread of zero. So is a negative `s2`, where the authors' published code returns the root of its magnitude. A cost sweep multiplies the estimate by its scale ([Cost sweeps](#cost-sweeps)).

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

With sizes on, an order's size is rounded down to the lot when it is placed, and so is every fill's size after its clamps, and a limit or a trigger is rounded to the instrument's price rule: a fixed tick, or Hyperliquid's five significant figures within its decimal cap, where a whole-number price is always allowed. An order below the minimum notional at the last close is refused, a reduce-only one exempt where the venue exempts it, and one given in the quote currency is judged by its amount. An order above the maximum order quantity is held to that maximum rounded down to the lot, and one given in the quote currency fills no more than that; a maximum below a lot is refused where the venue is built. The order log records a hold after the order's placement, and none for a placement it refuses; a value order's it records on a bar that prices its fill, before anything else of the order there, where the lots its amount buys at that price are more than that rounded maximum. A placement that leaves less than a lot to fill is refused, and an order retires when what remains of it rounds down to no lot. A replace is held to these as a placement, its size what it filled and the new size together. A bracket's legs, which close the position they find, take the price rule but not the lot or the bounds when they are placed. A position that floating-point sums leave below half a lot is zeroed at the bar's close with no fee: its value there, the holding's on spot and its unrealized profit on a perpetual, is recorded as a dust adjustment, and its round trip ends with the reason `dust`, its bracket with it. With no lot, a fill of 1e-9 base units or less, or of 1e-9 dollars of contracts on an inverse contract, moves nothing, and an order retires at that remainder.

```text
lot       q = lot * floor(q / lot + 1e-9)
tick      p = tick * round_half_even(p / tick)
figures   p = round_half_even(p * 10^d) / 10^d,  d = min(max(figures - 1 - e, 0), cap),
          where 10^e <= abs(p) < 10^(e + 1), e from -22 to 22
```

The `1e-9` keeps a size that floating point left a hair under a whole number of lots at that number, and a target's leg is rounded the same way toward zero. It can so carry a fill a billionth of a lot past what its clamps allowed: on spot the wallet takes that back ([Wallets](#wallets)), and on a perpetual it can leave the position that far past a cap or a tier's bound. `10^k` is exact for `k >= 0` and is `1 / 10^-k` below zero. The step of a price rule at a price is its tick, or `10^-d` under significant figures, taken at the limit's own price even where the grid below it is finer. The significant-figure rule is Hyperliquid's, whose perpetuals are linear: its cap is 6 decimals less the asset's size decimals on a perpetual and 8 less on spot, and its lot is `10^-szDecimals`. An inverse contract takes a tick, in dollars, its lot and maximum order quantity are in contracts and its minimum notional in the coin, and a limit or a trigger rounded to zero, which has no reciprocal, is refused.

**Worked example.** With a lot of 0.001, a buy limit for 0.2 on a bar whose budget allows 0.0756 fills 0.075, rounded down to the lot, and rests on for the remaining 0.125, and a buy for 0.0006 is refused when it is placed, since it rounds to nothing. A buy limit of 0.2004 at 99.404, under a tick of 0.01, rests as 0.200 at 99.40, the size and price its placed event records. With a minimum notional of 10 at a close of 150, a buy of 0.05 is refused, and a reduce-only sale of 0.05 is placed where the venue spares those; with a maximum of 1.0005, a buy of 5 is held to 1.000. Under five significant figures with a cap of one decimal, a limit at 99.987 rests at 100.0, where the cap leaves a step of 0.1, so under `through` a buy there fills on a low of 99.9 but not of 99.98; one at 123,456.7 rests at 123,457, a whole number. A holding of 0.0004 that floating point leaves under a lot of 0.001 is zeroed at a close of 250, and its 0.1 is recorded as dust.

<br>
<br>


## Accounting

### Fees

A fill of size `s` at price `p` pays `s * p * f`, at the maker or the taker rate of its instrument, in the settlement currency. An instrument charges the venue's maker or taker rate unless it has its own for that side, as a venue's spot and perpetual markets charge apart, and every rule that charges or sets aside a fee reads the rates of the instrument it trades. A round trip carries its entry fee and its exit fee exactly once: a closing fill's fee plus its share of the entry fee still standing on the position.

**Worked example.** A universe holds BTC on spot at maker and taker rates of 0.1%, beside BTC on a perpetual at the venue's 0.02% and 0.05%. A taker buy of 0.5 BTC on spot at 60,000 pays 30.00, and a taker sale of 0.5 on the perpetual at 60,000 pays 15.00.

### Positions

A fill that crosses zero closes the old side at its average entry, realizing the PnL on that part, and opens the remainder at the fill price. A same-side add volume-weights the entry. Cash moves only with the position, and money is float64, updated in event order, a fill's fee before the cash its trade moves.

**Worked example.** A long of 1.0 at an average of 100 sells 1.5 at 110: the long closes, realizing 10.00 before fees, and a short of 0.5 opens at 110.

### Round trips

A round trip is a maximal run of one sign on one instrument: it opens when a position leaves zero and ends when it returns to zero or flips, and resizing inside it does not end it. Its net PnL is the PnL between its fill prices less its fees and the funding charged on it while open. One still open at the end is reported, marked at the last close and flagged. It reports its instrument and side; its size, the largest position it reached; its entry and exit, the average prices of its opening and its closing fills; the bars and times it opened and closed on; its fees, every fee of its fills, with a flip's fee split by size between the round trip it closes and the one it opens; its funding; its net PnL; and its return, the net PnL over its size times its entry. Its best and worst excursions count only the part of each bar the trade was in: the entry bar from the fill to the close, which after a fill at the open is the whole bar and after a fill inside the bar what it surely traded then: the fill, the extreme beyond it, which came after it ([What a bar cannot show](#what-a-bar-cannot-show)), and the close; and the exit bar from the open to the fill.

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

- The spot wallet holds cash and spot holdings. A buy is clamped to what its cash pays for, fee included, and a sell to its holding; it never borrows and is never liquidated. A fill that floating point leaves a hair below zero, as a cash clamp or a lot's tolerance can, by no more than a hundred-millionth of its notional, leaves the wallet at zero, the difference recorded as a negative dust adjustment.
- The margin wallet holds the collateral behind every perpetual, in cross margin.
- A transfer is applied at the start of the bar after it was decided, before any fill. A transfer out of the margin wallet is clamped to the cash the wallet holds, so unrealized profit cannot be moved, and so that the margin left still covers the cap's requirement; one out of the spot wallet is clamped to its cash; a clamped transfer is an order-log event. With the cap off nothing is required, so a transfer can take a margin wallet holding an unrealized loss to zero equity, at or past its trigger, and the account is then liquidated at the last marks, before the bar opens.
- Account equity is the sum of both wallets. An account whose equity is at or below zero is terminal.

**Worked example.** A spot wallet of 1,000 USDT buys 20 at an open of 100 with a taker fee of 0.1%: its cash pays for `1,000 / (100 * 1.001) = 9.99001`, so 9.99001 fills and the order log records the clamp. A margin wallet with 600 USDT of cash and 400 of unrealized profit, under a cap requirement of 200, moves at most 600 to the spot wallet, the cash it holds, though its equity less the requirement is 800. With the cap off, a margin wallet of 1,000 USDT holding a long of 10 from 100, marked at 95, moves at most 950, which leaves its equity at zero, and the long is liquidated at 95 before the next bar opens.

<br>
<br>


## Funding

### When an event is charged

- A row of the clock owns the funding events stamped after its open and up to and including its end, the open plus the interval, and they are charged at its closes after the liquidation check, on the positions still held. An account the check liquidates holds nothing at the closes and pays nothing there.
- An event stamped after one row's end and up to the next row's open, across a closure or a hole in the data, is charged at the start of the next row, before its gap segment, at the last marks, on the positions that carried through it. A debit that takes the account to its trigger there liquidates it at the start of the gap.
- An event stamped inside a row, which happens when a bar is longer than the funding interval or its boundaries miss the funding times, is charged at the row's close on the position after the bar's fills, and `intrabar` decides whether it counts toward the bar's liquidation, from the open, on the path carried in and in the check alike: under the pessimistic setting a debit counts and a credit does not. An event counted toward a check that liquidates is paid, and one not counted is not.
- A stamp within one second of a row boundary is taken as that boundary.

### The charge

At an event with rate `r` and price `p`, the margin wallet falls by `q * p * r` for each perpetual: a long pays a positive rate and a short receives it. The price is the instrument's close on the row that owns the event, or its last mark when it printed no bar there or the event fell between rows. Once the events are charged the account is checked at the closes, and if a debit has taken margin equity to its trigger, the forced close is booked at the closes, and book then backstop reads the closes as its whole path ([The forced close](#the-forced-close)).

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

Had the rate been 2%, the debit of 410.00 would leave equity of 90.00 under the requirement of 102.50, and the account would be closed at the close, 41,000: under takeover its remaining 90.00 is booked as the liquidation charge, and under book then backstop at two thirds a sale there pays the fee of 10.25 and leaves 79.75, above the backstop level of 68.33, which the trader keeps. Had the 0.05% event been stamped at 07:30, inside the bar, the pessimistic setting would count its debit from the open, which moves the trigger to 40,221.61, above the low of 40,210, so the account would be liquidated there and the debit paid. The path carried into the bar counts the debit too, so the bar is clipped at 40,221.61 before its orders resolve, and a stop resting at 40,215, which the low reaches, never fills. The optimistic setting would not count the debit, and the account would survive as before.

**Worked example across a gap.** The same account holds its long on an instrument whose bars stop at Friday's close of 41,000 and resume at Monday's open of 40,150, with an event at a rate of 0.05% stamped on Saturday. The decision at Friday's close sees equity of 500.00, because the event has not happened yet. At the start of Monday's bar the debit of 10.25 is charged at the last mark, 41,000, which moves the trigger to 40,221.61; the gap segment from 41,000 to 40,150 crosses it, so the account is liquidated on the gap at 40,221.61, where equity and the requirement are both 100.55.

<br>
<br>


## Margin

### The cap

- Summed over the perpetuals, each position's notional over the maximum leverage of the tier its size falls in may not exceed margin equity: `sum_i N_i / L_k(i) <= E`, with equity measured before the fill's own fee, but for a flip after the fee on the side it closes, so a flip opens no more than its close leaves room for. A leverage of 0.0 removes the cap for that instrument, and a fill past the last tier's bound is clamped there, unless the table leaves that bound open, at `inf`, as a venue that sets no largest position does.
- A growing fill is clamped to what the cap allows, which the order log records ([The fill log and the order log](#the-fill-log-and-the-order-log)).
- When one fill is checked, every other leg is marked at the same fraction of its own bar's move as the fill sits at on its own bar ([The path](#the-path)), or at its favourable extreme under `intrabar="optimistic"`. The fill is placed on its bar from the open to the adverse extreme of the side its instrument holds, so a flip sits where its old side leaves, and from flat of the side it opens.
- While the account is over the cap because equity fell, no fill may raise the summed requirement, and a flip must fit the cap outright. A position over the cap is never force-reduced. With margin equity at or below zero, only reducing fills pass.

**Worked example at one instrument.** A margin wallet of 10,000 USDT at 20x holds at most 200,000 of BTCUSDT notional, so a market buy of 5 BTC at an open of 60,000 fills 3.33333, and the order log records the clamp.

**Worked example.** A margin wallet of 10,000 USDT; BTCUSDT allows 20x and ETHUSDT 10x. A long of 1 BTC at 60,000 uses `60,000 / 20 = 3,000` of the cap.

- A market sell of 30 ETH at the next open, with BTC still at 60,000, has room for `(10,000 - 3,000) * 10 = 70,000` of notional, 23.333 ETH at 3,000. It fills 23.333, and the order log records the clamp.
- A sell limit on ETH that fills at its high of 3,300, on a bar where BTC falls to 54,000, sits at the end of its own adverse move, so BTC is marked at the end of its move too. Equity there is 4,000, BTC uses 2,700, and the room is 13,000 of notional, 3.94 ETH. Marked at its open instead, BTC would have left room for 21.2 ETH, which at BTC's 54,000 puts the account at 2.4 times its cap.

### Maintenance

- **Notional tiers** (Binance, Bybit, Hyperliquid): a position's requirement is `M_i = N_i * m_k - c_k` in the tier its notional falls in, and the account's is the sum. The cumulative amounts make it continuous across tier boundaries: `c_k = c_(k-1) + (m_k - m_(k-1)) * up_to_(k-1)`.
- **Quantity tiers** (OKX): the tier is chosen by the position's size in contracts, and its rate applies to the whole position.
- A size or a notional within a billionth of a tier's bound is at that bound, in the tier the bound closes, for the margin the cap asks of a position, its notional over its tier's leverage, and for a quantity table's maintenance, so lots that floating point sums a hair past a bound stay in its tier. A notional table's maintenance, continuous across its bounds, reads them exactly, so the liquidation path's segments meet them, and the cap reads the instrument a fill grows at its bounds exactly, so a fill it holds at a bound is held at the bound itself.
- The loader refuses a notional table whose amounts break continuity or whose first amount is not zero, any rate not below one, any tier whose maintenance rate reaches its initial rate `1 / L_k`, a table whose bounds do not rise, whose leverage rises or whose maintenance rate falls from one tier to the next, a leverage of 0.0 standing for no cap, since both requirements have to grow with the position, and a value that is not finite, but for a last bound left open.

**Worked example.** On a quantity table whose second tier runs to 200 contracts at 5x, fills of 107.1, 81.8 and 11.1 contracts of a lot of 0.1 hold 200, which floating point sums to 200.00000000000003: the position is in the second tier, at 5x, and a position of 200.1 is in the third.

**Worked example.** BTCUSDT's first two Binance tiers: up to 300,000 at 0.4% with no amount, then up to 800,000 at 0.5% with an amount of 300. At 300,000 of notional the first asks `0.004 * 300,000 = 1,200` and the second `0.005 * 300,000 - 300 = 1,200`.

**Worked example.** Hyperliquid keeps half the initial rate at a tier's maximum leverage and leaves its last tier open: BTC's table runs to 150,000,000 at 40x, then on at 20x, so its rates are 1.25% and 2.5%, and the second tier's amount is `(0.025 - 0.0125) * 150,000,000 = 1,875,000`. A position of 200,000,000 keeps `0.025 * 200,000,000 - 1,875,000 = 3,125,000`, and no bound holds it.

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

`x_i` is the low for a long and the high for a short. Equity is linear in `s` and the requirement piecewise linear, so the first crossing of any level is solved exactly, within each segment between tier breakpoints. On an inverse contract each segment runs in the reciprocal of the price, `1 / p_i(s) = 1 / a_i + s * (1 / o_i - 1 / a_i)` on the gap and likewise on the bar, so its equity is linear in `s` as well, and a fill's fraction on its segment is read the same way.

Legs that change during the bar join the path:
- a leg opened or grown during the bar enters at the fraction where its fill price lies on its adverse segment; a fill on the favourable side of the open enters at fraction 0, on its own segment from the fill price to the adverse extreme;
- a leg that flips is tested on both segments, the old side up to the flip's fraction on its own segment, measured from the flip's price since the wallet already holds what it realized, and the new side from there to its own adverse extreme; an old side that joined during the bar leaves no earlier than it joined, so a second flip on one bar never sets its new side beside the first flip's old side; a flip filled past the old side's adverse extreme traded on the gap, before the bar opened, so the old side leaves at the open;
- an instrument flat at the bar's start takes the segment of the side its first fill opens;
- an instrument unavailable on the bar keeps its last mark and does not move.

Under `intrabar="optimistic"` the account is liquidated on a bar only if every path inside the bars reaches the trigger. Every path starts the bar where the synchronous path does, so an account past the trigger there is liquidated there. Otherwise the test is one corner per perpetual held, that instrument at its adverse extreme and every other at its favourable extreme, and the closes. If none of them reaches the trigger, a path exists that never does, from the opens to every favourable extreme, out to each adverse extreme and back in turn, and on to the closes; rule 6 checks the closes again once the row's funding is charged, so a debit the bar did not count can still close the account there. If one does, every path does, and the clip falls where the synchronous path meets the trigger. A leg that a fill opens on the favourable side of its open still runs from its fill price to its adverse extreme, as under the default, though a path may have passed that extreme before the fill.

### The trigger, close and clip points

An account keeps maintenance where a leg on its path has a maintenance rate, a leg that left during the bar among them, and its forced close reads it as the path does.

- **The trigger point** is where margin equity falls to the maintenance requirement, `E = M`; with no maintenance, where equity reaches zero.
- **The close point** is where equity, less a closing fee on every position's notional, reaches zero: `E = sum_i f_i * N_i`, each at its instrument's liquidation fee when there is no maintenance, the taker rate unless the venue sets one, and at its taker fee when there is.
- **The clip point** is where the account leaves the market: with maintenance, the trigger point; with none, the point from which the account stays at or past the close level all the way to the trigger.

For one long position in one tier:

```text
trigger, with maintenance      p = (q * e - W - c) / (q * (1 - m))
close point, no maintenance    p = (q * e - W) / (q * (1 - f))
```

For a short, `(1 - m)` becomes `(1 + m)` and `(1 - f)` becomes `(1 + f)`, with `q` negative.

### The rule

1. Any funding stamped between rows is charged at the last marks, and the path is solved for the positions carried into the bar, with any funding stamped inside the bar counted on the bar segment as `intrabar` decides, since it comes after the open. If the trigger is not on the path, nothing is clipped.
2. If the clip point lies on the gap segment, its start included, the account was liquidated before the bar opened: the forced close is booked first, and no perpetual order fills on the bar. Funding stamped inside the bar is not paid, even where counting it put the trigger on the bar, since the account held nothing by its stamp, and book then backstop reads the bar segment without it. The spot wallet trades on.
3. Otherwise the bar of every perpetual the account holds is clipped at the clip point's fraction of its own move, and the bar's orders resolve against the clipped bars. A perpetual the account does not hold is not clipped. Spot bars are never clipped. For an account with maintenance, a fill during the bar that opens, grows or flips a perpetual, with orders left to resolve, has the path solved again for the account it leaves, as the check solves it, and where that path reaches the trigger every held perpetual's bar is clipped there for the orders left, so none fills past the point the account left the market; a clip only tightens. Without maintenance the trigger is the close point, past which the margin bound already holds every reducing fill.
4. Under the pessimistic setting, the bar's fills are first resolved in their order, each perpetual fill skipped that, when its turn comes, would reduce the position it meets with its reference, the price the bar reached for it, on the favourable side of where that position started on the bar: its open for a position carried into the bar or opened by a market order, the reference of the fill that opened or flipped it, or of a later fill inside the bar that grew it, for one a resting order opened or grew. The path takes a position away from where it started, so the bar reached such a price only on a move back, which it cannot show came before the trigger. The path is then solved for the account as it stands. If it reaches the trigger the fills skipped are cancelled; otherwise the bar resolves again with them, in the same order. A sale above the price at which the bar's own buy opened the long it would close is one of them, even below the open. Under the optimistic setting they resolve in slot order with the rest.
5. **The check.** The path is solved again against the raw bars for the account after the bar's fills, every carried leg from its open and every leg opened or grown during the bar from its fill price, with any funding stamped inside the bar counted as `intrabar` decides. If the trigger is on the path, the forced close is booked at the clip point recomputed for those positions and that balance. An account already past it where its path starts closes each carried leg at its open and each new leg at its fill price.
6. Funding owned by the bar is charged at the closes, and an account a debit takes to its trigger is closed there. A margin wallet that fees have left below zero with no perpetual held, which can happen when the margin bound stops at an average entry, is set to zero, and the deficit is booked as a negative liquidation charge.

### The forced close

- **Takeover** (Binance, Bybit, OKX and every hand-built venue): every leg closes at its price at the clip point, inside its raw bar or on the gap segment. With no maintenance each close pays the liquidation fee on its notional, which at the close point leaves exactly zero. With maintenance the close pays no fee of its own. Whatever the margin wallet still holds, positive or negative, is booked as the **liquidation charge**, and the wallet ends at zero.
- **Book then backstop** (Hyperliquid): every leg is sold as a taker from its trigger price, paying spread, slippage, impact and the taker fee, and the trader keeps what remains. The price is held as any taker's is, inside the raw bar, and for a sale before the bar opened, which is priced off its trigger on the gap, inside what traded from the last close through the bar. Each instrument's impact is sized on all it sells, and no sale is limited by the liquidity budget. Takeover applies instead if the margin wallet the sale would leave falls short of the backstop level, the venue's stated fraction of the requirement at the trigger, or if the path also reaches that level, equity at that fraction of the requirement, which `intrabar` decides, since the bar cannot show that the sale finished first ([What a bar cannot show](#what-a-bar-cannot-show)).
- Under either model an instrument with a leg carried into the bar and one that joined during it is closed in a fill for each, the last taking what its position still holds.
- A liquidation cancels every resting perpetual order of the account. An empty margin wallet takes no growing perpetual fill. The spot wallet is untouched.

### The margin bound

1. A reducing fill is placed on its position's path at its fraction. A fill on the favourable side of its open, or on a bar with no adverse move, is at fraction 0, and so is a fill past the adverse extreme, which traded on the gap before the bar opened.
2. The account's close-level equity is evaluated with every other leg at the same fraction, or at its favourable extreme under `intrabar="optimistic"`, and with the fill's own fee in place of the closing fee on the size it closes. The closing fee is the one the account the fill leaves would pay, so a fill that closes the last perpetual with maintenance holds the rest to the liquidation fee.
3. If that equity is below zero, the fill is lifted toward the price where it is zero, or a purchase toward zero where no such price lies above it, never outside the bar the fill met, its instrument's clipped bar or the part of it after a fill inside it ([What a bar cannot show](#what-a-bar-cannot-show)), nor past its position's average entry. The lift the bar holds moves the fill's price and is recorded as its adjustment; the part it cannot hold, times the size closed, is paid into the margin wallet, net of the fee it would have added to a sale or with the fee it would have saved a purchase. Both are booked as a negative liquidation charge, so the price PnL stays measured from the fill's reference.

**Worked example.** A margin wallet of 50.5, a long of 10 at 100, a fee of 0.1%, and a market sell on a bar that trades at 95 throughout. The close point is where `50.5 + 10 * (p - 100) = 0.001 * 10 * p`, so `p = 949.5 / 9.99 = 95.045045`. The sale fills at 95, the bar's high, and `(95.045045 - 95) * 10 * 0.999 = 0.45` is booked as a negative liquidation charge, which leaves the account flat with a wallet of zero.

### The error and its direction

For linear contracts, margin equity less the requirement rises with a long leg's price and falls with a short leg's, at a fixed quantity. Every real path stays inside the box the bars define, and the worst point of that box is the corner where every leg sits at its adverse extreme at once, where the synchronous path ends. So the pessimistic setting can only err toward liquidating, and the optimistic setting only the other way. The exceptions, each named in a result's assumptions with its direction:
- an instrument unavailable on a bar is frozen at its last mark, so a liquidation on a hidden path is not seen (optimistic);
- liquidation in slices is not modelled (pessimistic for large positions);
- auto-deleveraging is not modelled (optimistic for winners);
- a flip that rule 4 holds back is held whole, the side it would open with it, so an exit of that side, which the bar may have reached only after the trigger, can fill (optimistic);
- funding stamped inside a bar counts toward the liquidation on the size the path holds, but is paid at the close on the size the bar's fills left, so a fill that shrinks a position escapes part of a counted debit, or forgoes part of a counted credit, each against its setting's direction;
- under the optimistic setting, a leg that a fill opens on the favourable side of its open runs to its adverse extreme, which the bar may have reached before the fill, in the check and for the orders that can only reduce it, so an account some path spares can be liquidated, and a stop some path never reaches can fill (pessimistic).

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
  - Under book then backstop the position is sold at 40,201.01 as a taker and pays the taker fee of 10.05, so on a venue that takes no clearance fee the trader keeps 90.45 less the sale's spread, slippage and impact. The backstop level, two thirds of the requirement, is where `0.5p - 20,000 = 0.0016667p`, 40,133.78, which this bar's low of 40,150 does not reach; a low past it would hand the account to the backstop under the pessimistic setting, while under the optimistic one the sale finishes first. A sale costing more than 33.50 would leave less than that level at the trigger, and takeover would apply.

### Worked for two instruments

A margin wallet of 10,000 USDT holds a long of 1 BTC and a short of 20 ETH, entered and marked at 60,000 and 3,000, with no maintenance and a fee of 0.05%. The cap holds: `60,000 / 20 + 60,000 / 10 = 9,000`. On the next bar BTC opens at 60,000 with a low of 54,000, and ETH opens at 3,000 with a high of 3,300, so the gap segment is empty.

- Along the path, at fraction `s`, BTC is at `60,000 - 6,000s` and ETH at `3,000 + 300s`, so equity is `10,000 - 12,000s`. The total notional stays 120,000, so the closing fee is a constant 60.
- The trigger: `10,000 - 12,000s = 0` gives `s = 0.8333`.
- The close point: `10,000 - 12,000s = 60` gives `s = 0.8283`, where BTC is at 55,030.00 and ETH at 3,248.50. With no maintenance this is the clip point.
- Both bars are clipped there and both legs close at those prices; equity there is 60, the fee is 60, and the wallet ends at zero.
- Had ETH charged its own taker rate of 0.1%, the closing fee would be `0.0005 * (60,000 - 6,000s) + 0.001 * 20 * (3,000 + 300s) = 90 + 3s`, and `10,000 - 12,000s = 90 + 3s` gives `s = 0.8256`, where BTC is at 55,046.24, ETH at 3,247.69 and the fees 92.48.

Under the optimistic setting, with BTC's high and ETH's low at their opens, the corners are BTC at 54,000 with ETH at 3,000, and ETH at 3,300 with BTC at 60,000, and the bars close at 57,000 and 3,150. Equity is 4,000 at each corner and at the closes, so the account survives the bar. Had the bars closed at 55,000 and 3,250, equity at the closes would be zero: every path ends there, so the optimistic setting liquidates too, at the synchronous path's clip.

<br>
<br>


## Inverse contracts

An inverse perpetual is quoted in dollars and settled in its coin. A contract is worth a fixed `V` dollars, so a position of `n` contracts, signed, holds `u = n * V` dollars, and everything the account counts is in the coin:

| Quantity | Linear | Inverse |
| :--- | :--- | :--- |
| profit from `e` to `p` | `q * (p - e)` | `u * (1 / e - 1 / p)` |
| notional | `abs(q) * p` | `abs(u) / p` |
| a fee at rate `f`, filled at `p` | `abs(q) * p * f` | `abs(u) / p * f` |
| a taker's cost at rate `c`, from its reference `r` | `abs(q) * r * c` | `abs(u) / r * c` |
| a funding charge at rate `r` | `q * p * r` | `u / p * r` |
| the average entry of fills `k` | `sum(q_k * p_k) / sum(q_k)` | `sum(n_k) / sum(n_k / p_k)` |
| the size a value order of `A` fills | `A / p` | `A / V` contracts |

- Every inverse quantity is linear in `1 / p`, so the tiers, the three points and the margin bound are solved as for linear contracts, in the reciprocal of the price, and each crossing is read back as a price. The tiers are by notional in the coin, or by size in contracts on a quantity table.
- The average entry is the harmonic mean of the fill prices, weighted by contracts, which keeps a position's profit `u * (1 / e - 1 / p)` equal to the sum of its fills' profits.
- What a fill paid beyond its reference is counted in the coin, `abs(u * (1 / ref - 1 / price))`, and its spread, slippage and impact share it in proportion to their rates.
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

One row per bar from the first, where row 0 is the opening balance, so no curve is one shorter than the data, and beside it the holdings, one row per bar and instrument, a bar's instruments in the Data's order.
- The ledger, per account: equity, each wallet's cash, gross and net exposure, leverage, margin used, the maintenance requirement, and that bar's fees, funding, spread, slippage, impact, liquidation charge and dust adjustment.
- The holdings, per instrument: position, average entry, mark, notional, unrealized PnL, weight, drift and pnl.

An instrument's pnl is what it has made since the run's first row: the cash its fills, its funding and the charges given to it have moved in the wallet it settles in, plus its position's value at the row's close, the holding's on spot and its unrealized PnL on a perpetual, nothing while it is flat. The instruments' pnl sum to equity less the opening equity on every row, within rounding. A takeover's liquidation charge is given to the instruments it closed, in proportion to each one's maintenance at the prices its legs were closed at, or to their notional there where none of them keeps maintenance; a margin wallet set to zero under rule 6 of [The rule](#the-rule) gives its deficit to the instrument whose fill or dust sweep last left no perpetual held.

**Worked example.** A margin wallet of 10,000 USDT holds a long of 1 BTC from 60,000 and a long of 20 ETH from 3,000, both at 20x, BTC keeping 0.5% maintenance and ETH 1%, and both bars fall 10% from their opens. At fraction `s` equity is `10,000 - 12,000s` and the requirement `900 - 90s`, which meet at `s = 9,100 / 11,910 = 0.764064`, with BTC at 55,415.62 and ETH at 2,770.78. Each leg's close there loses 4,584.38, and the 831.234 the wallet still holds, the liquidation charge, is split as each leg's maintenance there, 277.078 to BTC and 554.156 to ETH, so BTC's pnl is -4,861.461 and ETH's -5,138.539, which sum to the account's -10,000.000. Had both bars fallen 7% and closed at their lows, the path would not reach the trigger, and a funding debit of 1% at the closes, 558 on each leg, would take equity from 1,600 to 484 against a requirement of 837, closing both there; the 484 the wallet holds is below the requirement, and each instrument bears its share of it, 161.333 to BTC and 322.667 to ETH.

### The ledger identity

Each bar, the change in equity equals the price PnL less spread, slippage, impact, fees, funding, the liquidation charge and the dust adjustment. Price PnL is measured from reference prices, a fill's reference and then each bar's mark, and a fill's spread, slippage and impact are what it paid beyond its reference, each in proportion when the bar's range held its price back. Every term is computed on its own, so none is a residual. The identity holds within an error bound derived from the magnitudes involved. On a run that ends flat with no liquidation charge and no dust adjustment, the round trips' net PnL sums to the change in equity.

**Worked example.** A bar carries 0.5 BTC from a mark of 60,000 and buys 0.1 BTC. The reference is the open, 60,300. The buy pays an estimated half-spread of 3.00, a slippage of 1 bp (6.03) and the taker fee of 0.05%, for a fill at 60,309.03. The bar closes at 60,500.
- Price PnL: `0.5 * (60,500 - 60,000) + 0.1 * (60,500 - 60,300) = 250 + 20 = 270.00`.
- Costs: spread `0.1 * 3.00 = 0.30`, slippage `0.1 * 6.03 = 0.603`, fee `0.0005 * 0.1 * 60,309.03 = 3.0155`.
- Change in equity: `270.00 - 0.30 - 0.603 - 3.0155 = 266.0815`. From the fill directly, `250 + 0.1 * (60,500 - 60,309.03) - 3.0155` gives the same 266.0815.

### Costs

A run's costs are the totals of the ledger's cost columns over its rows: fees, funding, spread, slippage, impact, the liquidation charge and the dust adjustment, each a compensated sum, positive where the account paid it, in the asset the account settles in. They are measured from each fill's reference, as the identity is, so the move from a decision's close to the next open is price PnL, never a cost, and the price PnL less the costs is the change in equity.

**Worked example.** A margin wallet of 10,000 USDT on a linear perpetual, with a taker fee of 0.05%, a fixed spread of 1 bp, slippage of 1 bp, funding of 0.01% at every bar's close and volume far above every order. Row 0 trades at 60,000 throughout, row 1 opens at 60,000 and closes at 60,500, and row 2 opens at 60,400 and closes at 60,450. An array of quantities 0.1, 0 and NaN buys 0.1 at row 1's open and sells it at row 2's.
- The buy's reference is 60,000. Half the spread, 3.00, and the slippage, 6.00, put it at 60,009.00, for a fee of 3.00045, and row 1's close charges funding of `0.1 * 60,500 * 0.0001 = 0.605`, which leaves equity at 10,045.49455.
- The sale's reference is 60,400. Half the spread, 3.02, and the slippage, 6.04, put it at 60,390.94, for a fee of 3.019547, and the account is flat at row 2's close, where it pays no funding.
- The costs are fees 6.019997, funding 0.605, spread 0.602 and slippage 1.204, 8.430997 in all. The price PnL is `0.1 * (60,500 - 60,000) + 0.1 * (60,400 - 60,500) = 40.00`, so equity ends at `10,000 + 40.00 - 8.430997 = 10,031.569003`, as the round trip's net PnL says: 38.194 realized, less its fees and its funding.

### The fill log and the order log

- The fill log has a row per fill: time, instrument, order id, side, size and price; the reference price; spread, slippage, impact and fee; any adjustment that held the price inside the bar or applied the margin bound, and of it, per unit, what the margin bound lifted, so that the rest is the bar's hold; what the bound paid into the margin wallet for the part the bar could not hold; maker or taker; and the reason, one of `market`, `limit`, `stop`, `take_profit`, `rebalance`, `seam`, `liquidation` and `delist`. A forced close lifts nothing and pays nothing in.
- The order log has every order's events, each with its reason: `placed`, `armed`, `clamped`, `refused`, `not_filled`, `expired`, `cancelled` and `filled`.

A placed or armed event records the order as it is queued or rests: its side, its whole size, what it filled and what remains together, after the lot and the largest order, and a limit's price or a stop's trigger on the price rule. A bracket's leg that closes the position it finds has no size and, placed while flat, no side until it arms, and a value order on spot or a linear contract has no size, its reason giving its amount. A refused event records the side and the size the call gave and no price; a refused replace records the order's side and, where the call gave a size, what the order had filled with it; and a target's leg on an instrument yet to print is refused with the side of its weight and no size.

A clamp names the rule that held the fill. A fill's clamps apply in one order: the liquidity budget, then reduce_only, then on spot the wallet's holding for a sale or its cash for a buy, and on a perpetual the cap, which tests in turn for margin equity at or below zero, the last tier's bound and the summed requirement. The rule named is the last that reduced the size before the lot rounded it. A bar that printed no volume is named before every clamp, and a size below what a fill can book, such as a value order's amount that buys less than a lot, is named after them. A clamp that leaves part of a fill is a `clamped` event with the size it allows and the size asked. One that leaves nothing of a market order is that order's `not_filled` event, naming the rule; of a transfer, a `clamped` event of size zero; and of a resting order, no event, since the order rests on and would otherwise log every bar the rule holds it, so an ioc limit a clamp empties, or a fok limit one holds short of its size, expires after its bar for its time in force.

**Worked example.** A margin wallet of 10,000 USDT at 20x holds a long of 1 BTC at 60,000. A reduce-only market sell of 3 BTC fills 1, and its clamp names reduce_only, allowing 1 of the 3 asked; without reduce_only the sale would flip the account short by 2, which the cap allows. A reduce-only sell of 3 BTC with the account flat fills nothing, and its `not_filled` event names reduce_only, which allows none.

### Assumptions

A run lists each modelling choice it made as a row: its name, its value, its source, its grade, the direction of its error where it is simplified, and what it cost in the run where the run's records measure it. A choice switched off is listed as off, never left out.
- `venue`: the run does what a venue does, as far as bars show it.
- `simplified`: an approximation of what a venue does. Its direction names the way the rule leans on the events it decides, `pessimistic` where it refuses more, liquidates sooner or charges more, `optimistic` the other way, and `either` where it leans both ways; it promises no sign for a run's numbers.
- `optimistic`: an option that leaves out a charge, a cost or a limit every venue imposes, or decides what a bar cannot show in the trader's favour. A run with one is not a venue's, whichever way it moved that run's numbers, and only a row graded optimistic is an optimistic assumption.

A fixed rule of the engine is graded venue or simplified, never optimistic, since no setting moves it, and a dated table is graded venue on any bars, its source's date showing its age. Each row describes the rule each instrument ran: a tier table comes before a flat leverage, the price rule decides whether `through` applies, maintenance decides whether the liquidation fee is used, and leverage, maintenance and the liquidation fee are read over the perpetuals alone. Where instruments differ, the value names those that depart and the row takes the least favourable grade among them. A row reads values, not the option's word, so a fixed spread of 0 bp, a slippage of 0 and an impact of `k = 0` are not modelled and a funding rate of 0 is not charged, and its value reads after its name, as in "limit fills on touch".

| Row | Value | Grade | Direction |
| :--- | :--- | :--- | :--- |
| balance | each wallet's, "10,000 USDT" | venue | |
| fees | "maker 0.02%, taker 0.05%", each kind's on a venue of several | venue | |
| fees | "not charged", where a taker rate is at or below zero | optimistic | |
| leverage | "each instrument's tier table" | venue | |
| leverage | "capped at 10x", where a perpetual has no table | simplified | either |
| leverage | "not capped", where a perpetual has no table and a leverage of 0, or a tier of leverage 0 | optimistic | |
| maintenance | "each instrument's tier table" | venue | |
| maintenance | "not kept", where a perpetual keeps none | optimistic | |
| liquidation | "by takeover", "on the book, then by backstop at 2/3" | venue | |
| liquidation fee | "unused, as maintenance is kept" | venue | |
| liquidation fee | "the taker rate", "0.5%", where no perpetual keeps maintenance | simplified | either |
| liquidation fee | "not charged", where no perpetual keeps maintenance and the fee is zero | optimistic | |
| max open orders | "8 an instrument", or fewer | simplified | pessimistic |
| max open orders | more than 8 | simplified | either |
| limit fills | "on touch", under `touch`, or `through` with no price rule | optimistic | |
| limit fills | "through, half a step past the limit" | simplified | either |
| window | "24 bars" | simplified | either |
| liquidity | "10% of the trailing volume a bar" | simplified | either |
| spread | "not modelled" | optimistic | |
| spread | "fixed at 1 bp" | simplified | either |
| spread | "estimated, which errs high on the most liquid perpetuals" | simplified | either |
| slippage | "not modelled" | optimistic | |
| slippage | "1 bp" | simplified | either |
| impact | "not modelled" | optimistic | |
| impact | "sqrt, k = 0.1", "linear, k = 0.1" | simplified | either |
| sizes | "on the lot and the price rule" | venue | |
| sizes | "not rounded or bounded" | optimistic | |
| order limits | "a minimum notional and a maximum size" | venue | |
| order limits | what is missing, "a maximum size, no minimum notional", "none given" | simplified | optimistic |
| intrabar | "pessimistic" | simplified | pessimistic |
| intrabar | "optimistic" | optimistic | |
| intrabar | "random" | simplified | either |
| funding | "from the data" | venue | |
| funding | "not charged", with a perpetual in the run | optimistic | |
| funding | "every 8 hours at 0.01%", "every 3 bars at 0.01%" | simplified | either |

The rows only a perpetual has, leverage, maintenance, liquidation, the liquidation fee and funding, read "no perpetual", graded venue, where the run holds none, and order limits reads "sizes are off", graded venue, where sizes are off, since the sizes row then carries the grade. The engine's own rules follow, their source "the engine".

| Row | Value | Grade | Direction |
| :--- | :--- | :--- | :--- |
| margin | "cross, every perpetual on one wallet" | venue | |
| spot borrowing | "not modelled, a spot buy held to the cash" | venue | |
| liquidation trigger | "the bar's trade prices, not a mark" | simplified | pessimistic |
| funding price | "the close, not the mark" | simplified | either |
| fills in the bar | "a taker's price held inside the bar's range" | simplified | optimistic |
| margin bound | "a reducing fill held where the margin runs out, what the bar cannot hold paid in" | venue | |
| an instrument with no bar | "held at its last mark" | simplified | optimistic |
| liquidation in slices | "not modelled, which errs against large positions" | simplified | pessimistic |
| auto-deleveraging | "not modelled, which flatters winners" | simplified | optimistic |
| a flip held back | "held whole, the side it opens with it" | simplified | optimistic |
| funding paid inside a bar | "paid on what the bar's fills left" | simplified | either |
| funding counted inside a bar | "counted on the size the path holds" | simplified | against the setting |
| a leg opened on the favourable side | "runs to its adverse extreme, under the optimistic setting" | simplified | pessimistic |

An engine rule that cannot act on the run reads its off form, graded venue with no direction and no cost, the first that applies in this order. On a run with no perpetual, margin, the liquidation trigger, the funding price, the margin bound, liquidation in slices, auto-deleveraging, a flip held back, a leg opened on the favourable side and both funding rows inside a bar read "no perpetual", and an instrument with no bar keeps its value, graded venue, since what it hides is a liquidation; spot borrowing reads "no spot instrument" where the run holds none. With funding not charged, the funding price and both funding rows inside a bar read "no funding charged". A flip held back reads "none, under the optimistic setting" under that setting, and a leg opened on the favourable side "no exception under the pessimistic setting" under that one, while under `random` both keep their values. The two funding rows inside a bar read "no event fell inside a bar" where none did, and otherwise give the count after their value, "counted on the size the path holds, 730 events inside a bar". An event is inside a bar where it is stamped after a row's open and before its end on the run's own rows, a stamp within a second of either taken as that boundary, so neither an event between rows nor an `every_bars` event, charged at a close, is ever inside. Funding counted inside a bar errs against its setting: optimistic under the pessimistic setting, pessimistic under the optimistic one, and either under `random`.

The Data's rows follow, their source the Data's own: "the router, fetched" with its time, "a frame", "an array" or "joined Datas", with "resampled from 1h" after it where it was resampled.

| Row | Value | Grade | Direction |
| :--- | :--- | :--- | :--- |
| bars | "1h, 2023-01-01 00:00 to 2026-08-31 23:00, from binance", the router's bars of the run's venue | venue | |
| bars | the same, from a frame, an array or another venue's router bars | simplified | either |
| gaps | "none", "3, the longest 2 days" | venue | |
| funding events | "4,383", "none" | venue | |
| history | "from each instrument's first bar", or where the router's begins | venue | |
| delisting | "no delisting time; the router serves none", where every instrument's router bars run to the last row | venue | |
| delisting | "no delisting time; every instrument's bars run to the last", where bars that are not the router's do | venue | |
| delisting | "no delisting time; ETHUSDT's bars end 2025-03-01 12:00", where an instrument's bars end before the last row with no delisting time | simplified | either |
| delisting | "ETHUSDT delisted 2025-03-01 13:00", each delisting time the venue gives, and "ETHUSDT delisted 2025-03-01 13:00, after the last row" where no row the run saw reaches it | venue | |
| delisting | "ETHUSDT delisted 2025-03-01 13:00; no delisting time otherwise; SOLUSDT's bars end 2025-02-01 08:00", where another instrument's bars end before the last row with none | simplified | either |
| universe | "the user's list of 3 instruments" | simplified | optimistic |
| prices | "as the venue printed them", the router's bars | venue | |
| prices | "as given; splits, dividends and borrow costs only as the user adjusted them" | simplified | either |

A Data's bars are the router's where every instrument names its venue, and they read as another venue's where the run's venue is a preset of a different exchange. The delisting row's source is the venue's for the delisting times it gives, read as a venue row's is, and the Data's for bars that end with none.

A venue row's source is the input that decided each instrument's value, read key by key in the order a venue resolves an instrument's settings, its defaults, then its kind's table, then its own: a value equal to the preset's reads as the preset and its date, "binance_perp, taken 2026-10-16"; one the preset leaves out that equals the plain venue's reads "plain"; any other reads "set by the user". A venue built by hand is compared with the plain venue alone. Leverage and maintenance are sourced from an instrument's tiers where it has a table, and the liquidation fee from them where they keep maintenance, order limits the router gave read "the router, fetched" with its time, and a preset's slippage, impact and liquidity read as the preset, "a starting calibration", where they depart from the plain venue's. A row with one source reads it; otherwise it reads the source of the form its value leads with, then each key or instrument another source decided, "binance_perp, taken 2026-10-16; taker set by the user".

A row's cost is read from the run's ledger and fill log, and nothing is run again; below the full level, from its ledger and its round trips alone ([Recording levels](#recording-levels)). The fees row costs the fees column less the fees a takeover paid, and the liquidation fee row those fees. Funding, spread, slippage and impact cost their columns. The margin bound costs minus what it gave, the price it lifted times each fill's size, or on an inverse contract the coin that lift gave its contracts, plus what it paid in, and the liquidation row costs the liquidation column less the margin bound's part, which leaves it the takeovers' charges, the deficits set to zero and the rounding a fill left. The sizes row costs the dust column, and every other row has no cost. At the full level the rows with a cost sum to the costs' total within rounding, and a row's cost is its column's share as stated, which can hold another row's part: a sale on the book pays fees, spread, slippage and impact into their columns. The intrabar row is priced by running again with the other setting, which the user does.

**Worked example.** A run on binance_perp has no optimistic assumption. Moved to limits that fill on touch and no spread, it has two, "limit fills on touch" and "spread not modelled", each sourced "set by the user". In the one-instrument example without maintenance ([Worked for one instrument](#worked-for-one-instrument)), the liquidation fee row costs 10.005, the fee the close paid, and in the margin bound's example the margin bound row costs -0.45 and the liquidation row nothing.

### Baselines

A run recorded at the full level is set beside three baselines on its own data and venue, each a value per row of the run.
- **cash**: the opening equity on every row.
- **buy and hold**: an equal share of the opening balance `E0`, `E0 / N`, in each of the Data's `N` instruments, bought as the quantity it buys at the instrument's first printed close, or on an inverse contract the contracts it is worth there, `E0 * p / (N * V)` at a close of `p`, and asked for at every close from then on as an array of quantities is, on the run's venue and paying its costs. A leg the budget, the largest order or a bar with no volume cuts short is so completed at later closes, and none is ever rebalanced. An account of two wallets splits its opening balance between them in the universe's proportion for this, so every leg is one `N`-th of it and unlevered, and an instrument with no close above zero before the Data's last row holds its share in cash.
- **frictionless**: the same policy on the run's venue with the costs of trading removed and every rule kept. Maker and taker fees, the liquidation fee, spread, slippage, impact and funding are set to nothing, while the tiers, maintenance, the liquidation model and its backstop, the budget, the window, the limit rule, sizes and their bounds, `intrabar` and the balance stay as they were. It still pays what a forced close takes, so its gap to the run is the costs with what they changed along the way, not a split of them, and it is not the frictionless preset, which lifts the venue's limits as well.

An array runs again with its rows, sizing, tolerance and barriers. A strategy runs again on a copy taken when the run was called, played once, which shares with the run whatever the strategy does not hold itself, its class, its module's state and the global random state, as they stand when the baselines are computed, and they are computed when first read. A run the caller stepped has no policy to run again, and its frictionless baseline is missing. Buying and holding on spot asks again for the remainder a fee or an open above the decision's close left, which fills nothing; a position a forced close took is bought back where its wallet allows; and an instrument that lists late is sized on the opening balance's share. On an inverse account the long holds twice the coin's move in dollars, and a fall of about `(1 - m) / 2` liquidates it, `m` its maintenance rate, sooner as its funding draws the wallet down.

**Worked example.** In the costs example ([Costs](#costs)), cash reads 10,000.00 on every row. Buying and holding buys `10,000 / 60,000 = 1/6` BTC at 60,009.00 for a fee of 5.00075 and pays funding of 1.008333 and 1.0075, so it reads 10,000.00, 10,075.82 and 10,066.48. The frictionless baseline buys at 60,000 and sells at 60,400, reading 10,000.00, 10,050.00 and 10,040.00, and its last row's gap to the run's, 8.430997, is the run's costs. The rest charge no spread, slippage or impact, on volume far above each order.
- On a linear perpetual whose closes are 100, 104 and 99 and whose opens are 100, 101 and 103, with a taker fee of 0.1%, buying and holding with 10,000 USDT buys 100 at 101 for a fee of 10.10 and reads 10,000.00, 10,289.90 and 9,789.90. On spot the cash clamp buys `10,000 / (101 * 1.001) = 98.910990`, reading 10,000.00, 10,286.74 and 9,792.19, and the rest of the 100 it asks is asked again at the next close and fills nothing.
- On a linear perpetual with no charges, A opens at 100, 101, 103 and 98 and closes at 100, 104, 99 and 102, and B prints from row 2 at a close of 50, then opens at 51 and closes at 52. Buying and holding with 10,000 USDT buys 50 of A at 101 and, on the opening balance's share, 100 of B at 51, and reads 10,000, 10,150, 9,900 and 10,150.
- In the one-instrument example with maintenance ([Worked for one instrument](#worked-for-one-instrument)), a long the run and its frictionless baseline both carry into the bar is closed at 40,201.01 in each. Under takeover both lose the 100.50 the wallet holds there, and under book then backstop the baseline keeps all of it, where the run keeps 90.45 less the sale's spread, slippage and impact.
- On an inverse contract of 100 dollars at 0.4% maintenance with no charges, buying and holding with 1 BTC from a close and an open of 60,000 holds 600 contracts, which a takeover closes at `60,000 * 1.004 / 2 = 30,120`, and the baseline reads zero from then on.

### Tables

A run hands each record over as a table of named columns: the ledger, the holdings, the fill log, the order log, the round trips, the assumptions and the baselines. Every bar counts from the run's first row, so a row's time is the run's time at its bar, and every instrument is its name. A value a record does not have is missing, never a number that stands for none: a float the run leaves NaN, such as a drift no target has named, a fill's order on a forced close, an event's order, instrument, side, size and price where it has none, a round trip's close and its reason while it is open, its exit until a fill has closed some of it, an assumption's direction and cost where it has none, and a baseline the run cannot give. A missing float holds NaN under it and a missing integer or time the least int64, numpy's NaT, so neither reads as a value.

**Worked example.** An account trades BTCUSDT and ETHUSDT over three hourly rows, targeting a long of BTCUSDT at the first close and holding it to the end.
- Its ledger has three rows, at bars 0, 1 and 2, and its holdings six, bar 0's BTCUSDT and ETHUSDT, then bar 1's and bar 2's.
- No target names ETHUSDT, so its drift is missing on every row; BTCUSDT's is missing at bar 0 and holds from bar 1, the bar after the target was decided.
- Its one fill lands at bar 1 and names its order. Its round trip is still open at the end, so its close bar, close time, exit and reason are missing.

### Recording levels

| Level | Keeps |
| :--- | :--- |
| `"full"` | everything |
| `"account"` | the ledger and the round trips, with no holdings, fill log or order log |
| `"transitions"` | the account level's records and each step of an agent's episode, what offline reinforcement learning reads ([Transitions](#transitions)) |
| `"none"` | the final state only; `backtest` refuses it |

A run's statistics read only its equity, its closed round trips and the engine's own counts of its fills, its funding and its bars in a position, so at `account` they are the full level's to the bit, as its ledger and its round trips are. A run at `account` has no baselines and keeps no copy of its policy, and it reads its assumptions as it ends, pricing what its ledger and its round trips measure: where no round trip closed by a liquidation, the fees row costs the fees column and the liquidation fee row nothing, as at `full`, and where one did, neither has a cost; the margin bound and liquidation rows have none on any run. A strategy runs at `full`, since its account reads the order log, and keeps the account level's records when it ends. A search's trials run together in Rust keep their equity and their round trips alone. A validation's paths run at `full` and are not kept: each window and each path is scored from those records as its run ends.

**Worked example.** The run of [Costs](#costs) at `account` keeps its three ledger rows, its round trip and the statistics it has at `full`. Its fees row costs 6.019997 and its liquidation fee row 0.0, as at `full`, and its margin bound and liquidation rows have no cost, where at `full` each costs 0.0.

### Combining books

A strategy with legs at several venues is an account at each, each margined on its own, and its book is their sum; moving capital between them is not modelled. The book's clock is every time any account has a row, on one interval's steps. On a row of the clock each account counts its equity at its last row at or before it, its opening balance before its first row and its last equity after its last, and the book's equity is

`E(t) = E_1(t) + E_2(t) + ... + E_n(t)`,

added in the order the accounts are given. The book's returns and statistics are those of `E`, annualised as its accounts are where every one with an annualisation agrees, and otherwise at its rows less one per 365 days of the time some account covers, so a gap between two accounts' spans counts for neither. A row is in the market where an account holds a position inside its own span, and the book is bust where `E` reaches zero.

The book's ledger is its own, a row per row of its clock: each account's cash and its equity in the quote held as its equity is, an inverse account's equity in the quote before its first row valued at that row's close, its exposure, margin used and maintenance inside its own span alone, each cost on the row it was paid on, and the leverage gross exposure over `E`. Its holdings, fills, orders and round trips are the accounts' own, each row naming its venue and every bar counted on the book's clock, so a venue and an order identify one, and a bar two of its accounts were liquidated on counts once for each among its liquidations. Its assumptions are the accounts', in the order the accounts are given, each row naming its venue, and its costs are its ledger's. Each of its baselines is the accounts' summed as its equity is, each held at its opening balance before its first row and at its last value after its end, and one an account cannot give is missing from the book on every row. Accounts kept in different assets are not added, but the dollar assets USD, USDT, USDC, USDD, TUSD, FDUSD, BUSD, DAI, PYUSD, USDE, USDP, USD1 and RLUSD are counted at par where the book names one of them its asset, and an inverse contract's account is added only to others of its coin, its equity in the quote being dollars where any other account's is its own asset; neither are accounts on different intervals, off one interval's steps, or at different risk-free rates.

**Worked example.** Two accounts at two venues, with no fees.
- A holds 10,000 USDT on hourly bars from 00:00 to 03:00 and buys 100 of a coin at the 01:00 open of 100. The coin closes at 101, 100.50 and 102, so A's equity is 10,000, 10,100, 10,050 and 10,200, and its round trip is open at its end.
- B holds 5,000 USDT on bars at 01:00, 03:00 and 04:00, its instrument printing nothing at 02:00. It buys 100 at the 03:00 open of 50, the coin closes at 49, and it sells at the 04:00 open of 51.50, so its equity is 5,000, 4,900 and 5,150.
- The book's clock runs from 00:00 to 04:00. B counts its opening 5,000 at 00:00, before its first row, and at 02:00, held from its 01:00 row; A counts 10,200 at 04:00, after its end. The book's equity is 15,000, 15,100, 15,050, 15,100 and 15,350, a total return of 2.33%.
- A position is held on three of the four rows after the first: at 04:00 B is flat and A has ended. B's fills land on the book's bars 3 and 4, its own bars 1 and 2.
- A annualises at `3 * 365 / (3 / 24) = 8,760` and B, short of a row, at `2 * 365 / (3 / 24) = 5,840`, so the book annualises over the four hours its accounts cover: `4 * 365 / (4 / 24) = 8,760`.

<br>
<br>


## Search

`search` runs a policy over a space of parameters on the bars before a holdout, scores each trial on one run there, and keeps every trial beside the winner.

### The holdout

- `holdout` holds out the Data's last rows: a share in `(0, 1)` of its rows, rounded half to even, a whole number of rows from 1, or a time, the rows from the first at or after it. Each side needs at least two rows an instrument printed. `holdout=0` holds nothing out, and so does `holdout=None`, which warns that every score is then in-sample.
- The in-sample bars are a Data of their own that holds no bar of the holdout: the rows before it, with the bars before the Data's first, its instruments and their names, the funding events those rows own and a copy of the Data's provenance, under the fingerprint of `data[:cut]`. An instrument the whole Data holds funding for is funded in it, so a perpetual first funded after the cut runs as it runs in `backtest(policy, data[:cut], venue)`.
- The winner runs on the held-out bars once, as `data[cut:]`, so its warm-up falls inside them, and the look counts from the moment it is handed them.

**Worked example.** A Data of 1,000 hourly rows with `holdout=0.3` scores its trials on rows 0 to 699 and holds out rows 700 to 999, as `holdout=300` and the time of row 700 do.

### Trials

- A trial draws its parameters from the sampler, builds the policy with them and runs it once over the in-sample bars, with no Result. Its score is the statistic `objective` names, ranked as `direction` says.
- A trial is `complete` where it was scored, `floored` where it closed fewer round trips than `min_trades`, and `failed` where the policy or its run raised or its score was NaN. A floored trial ran, so it keeps its statistics and its returns and has no score, and a failed one keeps its error as text. An infinite score ranks.
- The winner is the complete trial with the best score, the lowest-numbered of equal ones. Every rule that picks one trial picks by its number, never by the order trials finished.
- A search with no complete trial raises: where its workers broke, naming them; else where a trial was floored, naming the most round trips any closed; and otherwise naming the last failed trial's error.

**Worked example.** With `min_trades=30`, trials that closed 12, 41 and 41 round trips at Sharpe ratios of 2.1, 1.4 and 1.4 are floored, complete and complete, and the winner is trial 1, the lower-numbered of the two scored 1.4.

### Samplers

- A space gives each parameter a range. `Int(low, high, step)` is the lattice `low + k * step` for `k` from 0 to `n = (high - low) / step`, and `Float(low, high, step)` the same lattice, its last point `high` exactly, or with no step the interval `[low, high)`. On a log scale an `Int`, of step 1, or a `Float` with no step is drawn evenly in its logarithm. `Categorical(choices)` is its choices.
- `"random"` draws each parameter in the space's order from the stream named `search random`: an `Int` or a stepped `Float` evenly over its lattice; an `Int` on a log scale as `exp(u)`, `u` drawn evenly on `[ln(low - 1/2), ln(high + 1/2))`, rounded half to even and held to `[low, high]`; a `Float` evenly on `[low, high)`, or on a log scale as `exp(u)` with `u` drawn evenly on `[ln(low), ln(high))`, held to `[low, high)`; and a `Categorical` evenly over its choices.
- `"grid"` runs the product of each parameter's values, an `Int`'s and a stepped `Float`'s lattice and a `Categorical`'s choices, the space's last parameter changing fastest; a `Float` with no step has none to list. Where the product holds no more points than `trials`, each point runs once in that order and the search runs that many trials; otherwise `trials` points are drawn without replacement from the stream named `search grid` and run in that order.
- `"tpe"` is Optuna's TPE with 10 startup trials, 24 candidates, the joint model, no grouping and the constant liar, seeded with the first draw below `2^32` of the stream named `search tpe`. A trial floored or failed is told to it as failed, and a stepped `Float` it proposes is moved to the lattice point of the nearest index, `round((v - low) / step)`. It chooses each trial from those told before, so it asks them in rounds of as many as there are workers and is told them in trial order.
- A seeded random or grid search repeats on any machine and at any number of workers. A seeded TPE search repeats at the same number of workers on the machine it ran on, since its kernels are numpy floating point.

**Worked example.** `Int(1, 10, step=3)` is 1, 4, 7 and 10, and `Float(0.01, 0.1, step=0.01)` is ten points from 0.01 to 0.1, the last exactly 0.1. A grid over that `Float` and `Categorical(["a", "b"])` is 20 points, (0.01, "a"), (0.01, "b"), (0.02, "a") and so on, and with `trials=100` the search runs those 20.

### The causality check

A search checks its policy for decisions taken on bars that had not closed. It runs trials one at a time until one has decided something at or after its warm-up on a row before its last printed one, and checks that trial before any other runs; before it returns, it checks the winner too, where the winner's parameters differ from that trial's.
- Each run of the check builds its own policy from the parameters, as a trial does.
- The policy decides on the whole in-sample bars twice, and where the two runs differ under the rule below, the search is refused, since what the check compares would not repeat.
- `w` is the first printed row the policy decides on: a strategy's first at or after its warm-up, an array's first holding a value. `P` is the printed rows from `w` to the one before the last printed row, `m` of them, and the five targets are `P[floor(k * m / 6)]` for `k` from 1 to 5, the k-th on instrument `(k - 1) mod N`. An array's decision on an instrument is its weight there, and a strategy's is the calls that name it, a target's weight for it among them, with the calls that name none. Each target moves to the first row of `P` at or after it where its instrument's decision changes to one that is not empty and differs from the one that instrument's cut before ended on; failing that, to the first where it changes to one that is not empty; and failing that, it stays. Each moved target is cut, and so is the row of `P` before it, where the decision it changes from held, so a lead that shows where a decision changes and one that shows where it holds both meet a cut's last row. A cut keeps the rows up to the next printed row after it, those a run plays at its close, coinciding cuts run once, and a cut holding fewer than two printed rows is dropped.
- A cut's decisions on every one of its rows, its last among them ([A strategy](#a-strategy)), must equal the whole run's there: the same rows holding none, each float within a bound of the other and every other value equal. A float64 array's bound is `1e-9` of its column's largest magnitude and a narrower float's 8 times its epsilon of it; a strategy's float arguments are held to `1e-9` of the larger of the two, or float32's bound where both are float32 values; and no bound is below `1e-12`.
- A difference is refused with a `LookaheadError` naming the cut and the first row, instrument and decision where the two differ, and so is a refusal of a cut's decisions, by the reader that takes a function's return or by a strategy's verb, chained from it; an error the policy's own code raises on a cut stays its own, with a note naming the cut, and one saying so where a strategy raised it deciding at the cut's last close. The check reads only the in-sample bars, so it is no look. A validation checks its policy in its first search, as a search does, then each winner its fits chose with parameters not yet checked, over the in-sample bars, and each path that ran, cut at each of its seams.

**Worked example.** A policy of weights `sign(close[t + 1] - close[t])`, computed with `numpy.roll`, decides each close from the next. On a cut ending at row `c` the roll puts the cut's first close at `c`, so its decision there is `sign(close[0] - close[c])`, and where that differs from the whole run's `sign(close[c + 1] - close[c])` the search is refused, naming the cut and row `c`.

**Worked example.** A policy of weight 0 on a close no lower than the four before it and the next one, and 1 elsewhere, reads the next close only to confirm a top, so a cut that ends on a top marks it as the whole run does. The cut on the row before that top ends on a rising close, the highest of five so far, which the cut marks a top and the whole run, seeing the next close higher, does not, and the search is refused there.

### Saving and resuming

- `save(path)` writes the Study into the directory `path`, made where it is missing:
  - `study.json`, the manifest: the format it is written in, the version that wrote it, the Study's kind, an id of the Study's own, which a Study loaded from it keeps, its provenance, each float there that is not finite as its text in an object of its own, its looks, whether its winner was handed the held-out bars and whether their Result is kept, its winner's number and whether a worker ran it, the trials its workers' breaking failed, a search's space, a model's pooled forecast or why it has none, and the rows its Data held, the row its holdout starts at, the first and last times of the bars before it and the balance its venue opened with;
  - a search's `trials.arrow`, its trials table, and `choices.arrow`, each trial's choice of each `Categorical` as its index among the choices, or a validation's `paths.arrow`, `fits.arrow` and `windows.arrow`, each an Arrow IPC file of its table;
  - `returns.npy`, where the Study kept its returns;
  - the winner's in-sample Result in `best`, run first where it has not been read, and its holdout's in `holdout` where it ran, each as `Result.save` writes one.
- It is refused where the directory holds files and none of `study.json`, `study.json.old` and `search.json`, since a save deletes or overwrites any file of a Study's there; where it holds a `study.json` or a `study.json.old` a save did not write; and where it holds the storage of another search, which the Study saved there would break.
- Every file is written under another name first and the manifest last, the manifest there before, or the new one where there is none, set aside as `study.json.old` until the new one is in place, so a save that an error stops leaves either the Study that was there or a directory `load` refuses until a save into it ends, its looks kept in the manifest set aside.
- `Study.load(path)` reads what `save` wrote, by its format. Refused are a directory with no manifest, one a save stopped in, naming it, a format this version does not read, and files that do not agree with one another. A loaded Study reads every table, statistic and summary the saved one did as it was saved; it holds no policy and no data, so `evaluate_holdout()` gives the saved holdout where there is one and is refused, before any look, where there is none.
- `search(..., storage=path)` keeps its Study in the directory `path`, made where it is missing. Into an empty one, a file a write stopped part way left under another name aside, it writes `search.json`, its settings and the looks counted at its held-out bars, before any trial runs; then each batch of trials as it arrives, a trial run alone, a chunk run together in Rust or a trial a worker gave back, as an Arrow IPC file of their rows and, where it keeps its returns, an `.npy` of them, named by the batch's first and last trial; and when it ends the Study, as `save` writes it, the batches then removed. A trial that failed because the workers broke is not written, and nor is the Study of a search whose workers broke, so the next call runs those trials, as a search resumed from that Study saved anywhere does; a Study whose winner runs otherwise than its trial did is not kept either, with a warning, and is returned as a search with no storage returns it.
- Into a directory that holds a search's settings or its Study, `search` resumes it:
  - It is refused where another version of probatum ran the search, named before its settings are read, since another version may name them otherwise; and where a setting differs from the stored one, naming the first in this order: the policy's name, the space, the objective and its direction, the seed, the sampler and its arguments, `min_trades`, the holdout as given, the rows it holds and the time they start at, `keep`, `risk_free`, the bars before the holdout, by their periods a year and their fingerprint, and the venue, and for TPE the number of workers; and where the directory's Study and its settings are of two searches. A seed of None takes the stored one. It is refused where `trials` does not reach past the highest-numbered trial stored.
  - Before any trial runs, the lowest-numbered stored trial that ran is run again, and the search is refused where any of its statistics differs from the stored one.
  - It runs the search's loop, in which each stored trial is taken in its turn rather than run, once the sampler asked for it gives the stored parameters, and is told to the sampler as it was, so the samplers' streams and TPE's history stand where they did and the causality check falls on the stored trial it checked. A stored trial whose parameters the sampler does not give is refused.
  - The trials not stored run as they would have, and the Study holds every trial in number order, as a search that never stopped does.
- A look is counted, before the winner is handed the held-out bars, in every directory the Study was saved or kept in that still holds its search: one more than the most counted there, written into each manifest there of its search, `study.json`, `study.json.old` and `search.json`, so neither a save stopped part way nor another Study of the search saved or kept there loses one, and a Study's looks are the most of its own and those counted where it was saved or kept. Where the directory holds this Study as it was saved, by the id its manifest names, the holdout's Result is written there once it ran, the manifest naming it only once it is in place. A resumed search keeps the looks counted where it is kept, and the stored holdout where its winner and its held-out bars are the stored ones; otherwise its winner's holdout is not yet run, and the looks before it still count ([The holdout](#the-holdout)). Where its final save stops part way, the next call keeps those looks but runs again the trials the Study it resumed held, and carries no holdout.

**Worked example.** A random search with `trials=100` into `storage` stops while trials 40 to 47 run, after the batch ending at trial 39 was written. Called again, it runs trial 0 again and compares it, asks the random stream for trials 0 to 39 and takes them as stored, and runs trials 40 to 99; its trials table and its returns are those of the same search run without stopping. Called once more with `trials=150`, it runs trials 100 to 149.

<br>
<br>


## Labels

`labels(data, venue, horizon, notional, stop=None, take_profit=None, side="long")` gives, for each row `t` of a Data and each of its instruments, what one trade of `notional` decided at the close of row `t` makes through the venue's fill model (ADR 0036).

- Each label is a fresh account on the Data's clock trading its instrument alone, at the venue, its balance the notional in the wallet the instrument trades from, opened at the close of row `t`. There it places a value order of `notional` ([Units](#units)), on an inverse contract one of the dollars `notional` coins are worth at that close, a buy for `side="long"` and a sell for `"short"`, which fills at the instrument's next open, what it cannot fill there cancelled.
- `stop` and `take_profit` are returns from the average entry, each a number or a `(T, N)` array read on row `t`, a NaN or None leaving that side without a barrier. The bracket is placed at the close of the bar the entry filled on, as an array run places one ([Barriers on an array](#barriers-on-an-array)).
- At the close of row `t + horizon`, or of the instrument's last printed row before it, a reduce-only market order closes what is left, filling at its next open, and another is placed at each close after while some is left and none waits.
- `exit` names what left the account flat: "stop", "take_profit", "horizon", "delist" or "liquidation", a position swept as dust taking the exit of the fill that left it. `end` is the row of that fill, `bars` is `end` less the row the entry filled on, and `ret` is the account's equity at the close of row `end`, less the notional, over the notional, net of every fee, funding payment, spread, slippage and impact.
- A row has no label where its instrument did not print at `t`, `t` is its last printed row, or it is delisted there, and where its entry filled nothing by row `t + horizon`: its `ret` is NaN, `exit` "", `bars` 0 and `end` the least int64.
- Any other label is censored where `t + horizon + 1` is past the last row, its trade never run, even where a barrier would have filled first, so censoring never depends on the outcome, and where its trade is still open at the last row: its `ret` is NaN, its `exit` "censored", its `bars` 0 and its `end` the larger of `t + horizon + 1` and `T`.
- `weight` is each label's average uniqueness among the labels of its instrument that have a return (Lopez de Prado 2018, chapter 4): a label spans the rows from the one its entry filled on to its `end`, `c_s` counts the labels spanning row `s`, and its weight is the mean of `1 / c_s` over its span. A row with no return has a NaN weight.
- `horizon` is a whole number from 1, `notional` a positive finite number, and `side` "long" or "short", which a spot instrument refuses.

**Worked example.** A frictionless venue, `notional=10_000.0` and `horizon=2`. The label at row 0 buys 100 at row 1's open of 100.0 and closes at row 3's open of 103.0: `ret = 0.03`, `exit` "horizon", `end = 3`, `bars = 2`. With `stop=0.05` the bracket rests its stop at 95.0 from row 1's close; row 2 opens at 99.0 and trades down to 94.0, so the stop fills at 95.0 on row 2: `ret = -0.05`, `exit` "stop", `end = 2`, `bars = 1`. On 10 rows the labels at rows 7 and 8 are censored, and row 9, the last, has none.

**Worked example.** Three labels of one instrument span rows 1 to 3, 2 to 4 and 3 to 5. `c` is 1, 2, 3, 2 and 1 on rows 1 to 5, so their weights are `(1 + 1/2 + 1/3) / 3 = 0.6111`, `(1/2 + 1/3 + 1/2) / 3 = 0.4444` and `(1/3 + 1/2 + 1) / 3 = 0.6111`.

<br>
<br>


## Validation

`validate` measures how a policy holds up across folds: a Strategy class or a function, searched over its parameters, or a model, fitted to labels. A searched policy runs a search over the bars before the holdout, whose winner the holdout runs, then the same search on each split's training bars, and trades each split's winner on its test span; a model is fitted to the same bars ([A model](#a-model)). The test spans join into out-of-sample paths, each one run of the engine.

### The splitters

- A splitter divides `T` bars. Its rows are the long, time-major rows an estimator reads, `T * N` of them, row `r` bar `r // N` of instrument `r % N`. `end`, where given, is each row's label end, an int64 array `(T,)` or `(T, N)`: the bar whose price closed the row's trade, from the row's own bar plus one on, at `T` or later for a label that runs past the bars, censored or running into a holdout, and the least int64 for a row with no label. Without `end` each row is a bar of one instrument holding a label of no length.
- `Holdout(tail)` tests the last `tail` bars, a share in `(0, 1)` of them rounded half to even or a whole number of them, and trains on the bars before.
- `WalkForward(train, test, anchored=False)` makes `w = floor((T - train) / test)` windows from the first bar: window `j` tests bars `train + j * test` to `train + (j + 1) * test - 1`, the last running to `T - 1`, and trains on the `train` bars before its first test bar, or with `anchored` on every bar before it. Bars appended move only its last window.
- `PurgedKFold(k)` divides the bars into `k` groups in order, the first `T mod k` holding `floor(T / k) + 1` bars and the rest `floor(T / k)`, as scikit-learn's KFold divides samples; split `i` tests group `i` and trains on the other bars.
- `CombinatorialPurged(groups, test_groups)` divides the bars into `groups` groups the same way, and its splits test every combination of `test_groups` of them, in the order `itertools.combinations` lists them, each training on the other groups. Its paths number `C(groups - 1, test_groups - 1)`, and path `p` takes group `g` from the `p`-th split, in that order, that tests `g`. `CombinatorialPurged(k, 1)` gives `PurgedKFold(k)`'s splits.
- `split(X)` refuses an X whose rows, `X.shape[0]` or else `len(X)`, are not `end`'s, checks every split, and gives each as `(train, test)`, ascending int64 positions of X's rows. It ignores `y`, and warns where `groups` is given. `get_n_splits` counts the same splits; a WalkForward built without `end` needs X to count them.
- A layout is refused where it leaves a group, a window or a side with no bar: `k` or `groups` above `T`, a `tail`, `train` or `test` of no bars, a tail of every bar, or a WalkForward with no window.

**Worked example.** 1,003 bars in five groups hold 201, 201, 201, 200 and 200 bars: 0 to 200, 201 to 401, 402 to 602, 603 to 802 and 803 to 1,002.

**Worked example.** `WalkForward(train=400, test=150)` on 1,000 bars makes 4 windows, testing 400 to 549, 550 to 699, 700 to 849 and 850 to 999; the first trains on 0 to 399 and, rolling, the last on 450 to 849, or anchored on 0 to 849. On 1,030 bars the last tests 850 to 1,029.

**Worked example.** `CombinatorialPurged(6, 2)` makes 15 splits, S0 testing groups 0 and 1, S1 groups 0 and 2, and so on to S14, testing groups 4 and 5, and 5 paths, which take groups 0 to 5 from these splits:

| Path | Groups 0 to 5 from |
| :--- | :--- |
| 0 | S0, S0, S1, S2, S3, S4 |
| 1 | S1, S5, S5, S6, S7, S8 |
| 2 | S2, S6, S9, S9, S10, S11 |
| 3 | S3, S7, S10, S12, S12, S13 |
| 4 | S4, S8, S11, S13, S14, S14 |

Every split appears twice: S0, S5, S9, S12 and S14 give both their groups to one path, and the other ten feed two.

### The purge and the embargo

- A split's spans are its runs of adjacent test groups, one span for WalkForward and Holdout. A span from bar `a` to bar `b` reaches `L`, the larger of `b` and the latest end among its rows that test, so a label censored or running into a holdout reaches nothing.
- A row tests where its bar is in a span and it has a label that ends before `T`. A row trains where its bar is among the split's training bars and it has a label that ends before `T`, unless `t < a <= e` for a span, its label reaching the span from before it, or `b < t <= L + E`, it starting before the span's reach or within the `E` bars of embargo after it. The intervals are closed, so a label that ends on a span's first bar is purged.
- `embargo=None` is the longest label held, the largest `e - t - 1` over the rows whose label ends before `T`, and 0 without `end`. It counts bars of the clock, and covers neither the features' memory nor the fill window. Holdout and WalkForward take no embargo, since no training row follows their spans.
- A split left with no row to train on or none to test on is refused, naming it, before any split is given.

**Worked example.** 1,000 bars whose labels at `t` end at `t + 25`, with `PurgedKFold(5, embargo=24)`. Split 2 tests 400 to 599, whose labels end as late as 624, so `L = 624`: the purge removes 375 to 399 and 600 to 624 and the embargo 625 to 648, and rows 975 to 999, whose labels end at 1,000 or later, never train. It trains on 0 to 374 and 649 to 974.

**Worked example.** `PurgedKFold(5, embargo=0)` on 1,000 bars whose labels at `t` end at `t + 11`, except the label at 590, which ends at 640. Split 2 tests 400 to 599, its last row's label ending at 610 and its latest at 640, so `L = 640`: the purge removes 389 to 399 and 600 to 640, rows 989 to 999 never train, and it trains on 0 to 388 and 641 to 988. A reach read off the last row would keep 611 to 640.

### Fits

- `validate` holds out the tail as `search` does ([The holdout](#the-holdout)) and lays the splitter's splits on the in-sample bars. Before the splits it runs a plain `search` over the bars the splitter gives the holdout as its test span, every in-sample bar or, with a rolling WalkForward, the last `train` of them, seeded from the stream named `validate final`; each split's search is seeded from the stream named `validate fit i`.
- A Strategy has no labels. A span from bar `a` to bar `b` reaches `L`, the first printed row after its last printed row, `b + 1` on a full clock, or `b` where it prints no row or none follows, since its last decision, at the close of its last printed row, is scored on the next. A split's training bars are the bars outside its spans, less `b + 1` to `L + E` after each span, `E` being `embargo` or 0.
- Each maximal run of training bars is a piece, run on its own. A piece that starts at the in-sample bars' first row runs on them, cut after its last bar, as a search's trial runs; a WalkForward window's piece runs on its `train` bars, or anchored on every bar before its window, as a slice of the in-sample bars. A piece after a span runs on a Data of its own holding the rows from `L + 1` to its last and no earlier bar, `L` being the latest reach of the spans before it, so no bar of theirs, nor one their last decisions are scored on, reaches its `init`, its `bar.history`, its fills' window or its funding; its account opens at the first printed row at or after `max(E, warmup)` rows into that Data, so no bar of its warm-up is scored, and at or after the row by which each instrument that printed before that Data has printed `W + 1` bars in it, or its last bar in it where it prints fewer, `W` being the venue's fill window, which counts an instrument's own bars and measures the first one's return from the bar before it: from there each fill reads the window the in-sample bars give it, as a fill of an instrument that printed nothing before that Data always does.
- Each piece of a trial plays its own copy of the Strategy built with the trial's parameters. A piece left with fewer than two printed rows from its opening adds nothing, and a split with no piece that can run is refused before any search.
- A trial on one piece is scored as a search's trial. On several, its statistics are the engine's over the pieces' equity joined end to end: each piece's curve scaled to continue from the end of the one before, the round trips' pnl as each piece made it, since each opens at the same balance, and the fills, the bars in a position and the funding summed. A join stops at the first piece that ends at or below zero equity, the later pieces adding their bars at that equity and nothing else. Every piece, fit, window and path is annualised as the in-sample bars ([Annualisation](#annualisation)). The objective is a statistic the order of the pieces does not change, so `max_drawdown_pct` and `calmar` are refused with PurgedKFold and CombinatorialPurged.
- A fit fails where its search has no complete trial; anything else a fit raises ends `validate`.
- The causality check runs in the first search, and then on each winner the fits chose with parameters it has not checked, over the in-sample bars, and on each path that ran, cut at each of its seams, so what `on_inherit` decides there is checked as `next`'s decisions are.

**Worked example.** `PurgedKFold(5)` on 1,000 hourly bars of a full clock, split 2 testing 400 to 599: the span reaches 600, so bar 600 is purged, and the piece after it runs on a Data whose first row is bar 601. With a fill window of 24 and a warm-up of 20, its account opens at its 25th printed row, 24 rows in, at bar 625; with a warm-up of 200, at bar 801.

**Worked example.** Two pieces of a trial each open at 10,000. The first ends at 11,000 with one round trip of pnl 1,000, and the second at 9,900 with one of -100. The join scales the second's curve by 1.1, so it ends at 10,890, a total return of 8.9%, and the round trips count as 1,000 and -100.

### A function

- A function `policy(data, **params)` validates as a Strategy class does, each trial on each piece the function called with the piece's Data and read as a search reads what it returns ([Trials](#trials)): weights run on the piece as an array run that opens where the piece's account opens, and a Strategy played there, the pieces joined as a Strategy's are. Every fit's trials decide in the kind the first search's winner decided in, weights or a Strategy, so a path is one kind of run, and a trial of the other kind fails, saying so.
- A path of weights is one array run over the in-sample bars ([Paths and seams](#paths-and-seams)). Each window's rows, from its first decision row to its last, are its fit's winner's weights, the function called with the in-sample Data, and its stop and take-profit those the winner returned, read on those rows; the rows before the first window hold. Every winner returns one sizing and one tolerance, and a path whose windows differ in them is refused, naming the windows. A path of Strategies is a Strategy class's, each window's Strategy the one its winner returns.

### A model

- `validate(model, data, venue, cv, features=, labels=, to_weights=, fill=None)` fits `model`, which has `fit` and `predict`, to `labels` on `features`. Before every fit a fresh estimator is made: scikit-learn's `clone(model)` for a model with `get_params`, or `model()` where a function of no arguments is given in its place, as a model whose `fit` resumes from its weights is given; any other model is refused, and so are `space`, `search` and `seed` beside one, which is fitted rather than searched. A function or a class given in a model's place beside `labels` is called with no arguments for each fit.
- `features` is a `(T, N, F)` array over the Data's rows or a function of a Data that returns one, read as `Data.features` reads it, computed once on the whole Data. A NaN on an available row past its feature's warm-up is filled where `fill` is given, as `Data.features` fills it, and otherwise leaves its row out of every fit and window, each fit counting its own. A function's features are computed a second time on the whole Data and refused where their in-sample rows differ from the first's, since what the check compares would not repeat; then on the Data cut after each of up to 10 rows spread evenly over the in-sample bars, and refused as a LookaheadError, naming the row, the instrument, the feature and the cut, where a value on a row up to the cut differs from the whole Data's. Values are compared as the causality check compares a float64 array ([The causality check](#the-causality-check)), a column's largest magnitude taken over its finite values and an infinity the same only as itself, and an error the function raises on a cut carries a note naming the cut.
- `labels` is a Labels of the same Data, refused where its rows, instruments or fingerprint differ. Its `end` on the in-sample bars is passed to the splitter, which is refused where it was built with `end` ([The purge and the embargo](#the-purge-and-the-embargo)), and its stop and take-profit are a path's barriers, each placed on the side of the position it guards, which `to_weights` decides.
- A fit reads, in the long, time-major order, its split's training rows that are available, have finite features and a finite `ret`, and are left by the purge and the embargo: `x` their features, `y` their `ret`, and their `weight` as `sample_weight` where `fit` takes that keyword. A fit with no row to read is refused, naming it.
- A window predicts on its rows, from its first decision row to the next window's, that are available with finite features, in the same order, one finite number a row; the predictions return to their rows of a `(T, N)` grid, NaN elsewhere, and `to_weights` of the window's rows of that grid gives its weights, of the same shape, a NaN holding, refused where they hold an infinity or short a spot instrument. A path is one array run over the in-sample bars, as a function's path of weights is, at `sizing="weight"` with the labels' stop and take-profit.
- Before the splits a final fit reads every in-sample row a fit could, but those whose label ends at or past the holdout's first row, and with a rolling WalkForward those before its last `train` bars. Its predictions on the in-sample rows and `to_weights` of them are computed a second time and refused where they differ, as a function's features are. Then its predictions on the rows up to each of the features' cuts, and `to_weights` of the in-sample rows' predictions up to it, the predictions first, are refused as a LookaheadError, naming the row, the instrument and the cut, where one differs from the whole's as a feature may not, or where they are refused on the cut and were not on the whole, chained from that refusal; an error the model's or `to_weights`' own code raises on a cut carries a note naming the cut. Its predictions traded on the in-sample bars, or with a rolling WalkForward on its last `train` bars as the first search trades a searched policy, are the Study's `best`, and `evaluate_holdout` trades its predictions on the holdout's bars, from the features computed on the whole Data.
- A path's forecast reads the rows its windows predicted that have a finite `ret` and a label that ends before the holdout, so no held-out bar prices it. Its rank correlation `rho` is Spearman's, the Pearson correlation of the predictions' and the returns' mid-ranks over those rows; its standard error is the square root of a Newey-West variance over the bars, Bartlett's weights at a lag `L` of the largest `end - t` among those rows, of each bar's sum of the rows' influence `u * v - rho * (u^2 + v^2) / 2`, `u` and `v` the mid-ranks less their mean over their deviation with the rows' count in its divisor, over the rows' count squared, the bars between holding none adding 0; and its R squared is `1 - sum((y - p)^2) / sum(y^2)`, against a forecast of zero. A forecast reading fewer than 3 rows, or predictions or returns that never vary, has none. The pooled forecast reads every complete path's rows.

**Worked example.** Four rows on four bars predict 0.01, -0.01, 0.02 and 0.0 where the labels returned 0.02, -0.01, 0.01 and -0.02: the mid-ranks are 3, 1, 4, 2 and 4, 2, 3, 1, so `rho = 3 / 5 = 0.6`, and the R squared is `1 - 0.0006 / 0.001 = 0.4`.

### Paths and seams

- A path's windows are its runs of groups, or its WalkForward or Holdout spans, that one split's winner trades; a group with no row a run decides at, a printed row before the last, makes no window. A window decides at the closes of its printed rows `f` to `l` and owns what those decisions govern: the returns, fills, closed round trips, bars in a position and costs of the account's rows from the one after `f` to the one after `l`, the last window's to the end.
- A path is one run, over the in-sample bars, of a composite Strategy whose account opens at their first printed row. Its `init` builds each window's Strategy with its fit's parameters, initialises each on the in-sample bars and reads each one's warm-up; each decides only at its own window's closes and from its own warm-up. A path's `warmup` is its first window's first row, so nothing trades before it.
- At the close of a window's first decision row, that bar having filled what the window before it left: every order not yet filled is cancelled, the waiting market and value orders in the order they were placed and then the resting ones by slot, each with the reason `seam`; every target is forgotten, so an instrument records no drift until a target names it again; the decision's events start empty, and the account's round trips and equity curve start from that close; `on_inherit(bar, account)` is called; and then `next`. A path's first window has no seam.
- With `seam="flatten"` the last decision row of each window but a path's last closes every position instead of calling `next`: each order not yet filled is cancelled as above, and a reduce-only `ioc` market order closes each instrument's position, in the instruments' order, placed and filled with the reason `seam`. What it leaves open, for want of liquidity, a size limit or a bar, is the next window's. Under `flatten` a window with another after it holds at least two printed rows.
- A path of weights, a function's or a model's, is one array run whose rows are its windows', and its seams are an array's. At the close of a window's first decision row every order not yet filled is cancelled with the reason `seam` and the targets are forgotten, as at a Strategy's seam; then each instrument holding a round trip places again at its average entry, at the returns of the row that opened the round trip, the legs of its bracket still resting, so a leg its partner's fill cancelled stays cancelled, and one that a barrier holds flat stays held; and then the row is read. With `seam="flatten"` the last decision row of each window but a path's last closes every position instead of being read, and at the seam after it a round trip the close left open places again the legs its bracket held resting until that close.
- A window's statistics are the engine's over its rows, opening at the equity of the close before them, and a path's over the rows from its first window's first decision on. A path whose run raises, or that a failed fit trades, fails, keeping its error, and `validate` raises where no path completes.

**Worked example.** Window 1 decides at the closes of rows 0 to 9 and window 2 from row 10. At row 9's close window 1 holds a long of 1 BTC with a sell stop resting at 95, and buys ETH, which prints no bar at row 10. Under `cancel`, at row 10's close the stop and the ETH buy still waiting are cancelled with the reason `seam`, window 2's `on_inherit` finds the long of 1 and nothing resting, and its `next` runs; window 1 owns rows 1 to 10 and window 2 rows 11 on. Under `flatten`, window 1 does not decide at row 9: its stop is cancelled there, and a sell of 1 BTC placed there fills at row 10's open with the reason `seam`, so window 2 inherits no position.

**Worked example.** A path of weights with `stop=0.05` whose window 2 starts at row 6. Window 1's weight of 1.0 at row 2 fills at row 3's open of 100.0, and its stop rests at 95.0 from row 3's close. At row 6's close the stop is cancelled with the reason `seam` and placed again at 95.0, and then window 2's row 6 is read. Had the stop filled on row 5, window 2's rows asking for a long would buy nothing until one asked for zero or less.

### The Study of a validation

- `paths`, a row a path: its number, state, score, the 15 statistics and the seven costs of the rows it owns, and its error. `fits`, a row a split: its number, what it tests, its training, purged and embargoed bars, its pieces, its winner's parameters and fitted score, its trials run and complete, its seed, state and error. `windows`, a row a path's window: its path and fit, the first and last account rows it owns and their times, the rows it decided on after its warm-up, its score, statistics and costs.
- A model's paths and windows are scored on the Sharpe. Its `paths` add each path's `rank_ic`, `rank_ic_t` and `r_squared` and `predicted`, the rows its forecast read; its `fits` count rows rather than bars, `train`, `purged` and `embargoed` among them, and hold `rows`, the rows the fit read, and `dropped`, those a NaN feature left out, its warm-up's among them, in place of the pieces, the parameters, the score, the trials and the seed; and the Study's `forecast` is the pooled forecast, a Forecast holding `rank_ic`, `error`, `t`, `lag`, `r_squared`, `rows` and `paths`, None for a searched policy.
- `returns` holds each path's returns on the in-sample account rows, `(paths, rows)`, NaN where the path owns no row and on a failed path, the columns a search's on the same bars; `keep="none"` keeps none.
- `best`, `best_params`, `best_score`, `evaluate_holdout`, `holdout` and `looks` are the first search's ([The holdout](#the-holdout)), and a model's its final fit's, its `best_params` empty and its `best_score` the Sharpe of its `best`.

<br>
<br>


## Inference

`probatum.stats` says how far a run's statistics can be trusted: an interval for each, the Sharpe's chance of beating a benchmark and the track record that would show it, the difference between two Sharpes on the same bars, alpha and beta against a benchmark, and what the order of the trades decided. It reads only what a run records, and every interval names its method.

### What a statistic reads

- A Result's series is its `n = T - 1` returns after the first row, `r_t = E_t / E_{t-1} - 1`, zero after a row whose equity is not above zero, which are the returns its statistics read, at its annualisation `ppy` and its risk-free rate, so a Result refuses `periods_per_year`. A plain series, a 1-D array, a list or a pandas Series read by position, needs `periods_per_year`, which has no default, and takes `risk_free` as an annual rate, 0 unless given; a NaN or an infinity in it is refused, naming its first position.
- The excess returns are `x_t = r_t - risk_free / ppy`, and the Sharpe is `sqrt(ppy) * xbar / s'`, `xbar` their mean and `s'` their deviation with `n - 1` in its divisor, as the record's is.
- An interval is computed when it is first read and then kept; a search's trials never compute one, so a trial's row holds point estimates. Its level is in `(0, 1)`, 0.95 unless given. Its bounds are in its statistic's units and sign, a drawdown and a value at risk positive magnitudes, which a text prints as losses.
- `autocorrelation(x, lag)` is `sum_{t > lag} (x_t - xbar) * (x_{t-lag} - xbar) / sum_t (x_t - xbar)^2`, 1 at lag 0 and 0 where the returns never vary or number no more than `lag + 1`, and `effective_sample(x)` is `n * (1 - a) / (1 + a)`, `a` the first autocorrelation held to `[0, 0.99]`, at least 2, and `n` itself below 4 returns. No interval reads either.
- A resampling draws at least 2 resamples, and an interval enough of them for its level that its two order statistics are two resamples; a smaller `resamples` or `n` is refused.
- Each resampling draws from a stream of its own, PCG64 from a NumPy `SeedSequence` of a 64-bit BLAKE2b digest of the float64 values it resamples, keyed by its name: `stats returns` on the returns, `stats trades` on the round trips' pnl, `stats sharpe difference` on the two series' returns, the first series' before the second's, and `stats trade order` on the round trips' returns. An interval is therefore the same on every read, after a save and a load, and in any process, and `seed=` stands in for the digest.

### The Sharpe's variance

For the excess returns, with `n` in every divisor: `mu = (1/n) * sum_t x_t`, `s^2 = (1/n) * sum_t (x_t - mu)^2`, `S = mu / s` and `u_t = (x_t - mu) / s`.
- The Sharpe's influence series is `z_t = u_t - (S / 2) * (u_t^2 - 1)`, the delta method over the joint moments of returns and squared returns, and `g_j = (1/n) * sum_{t > j} z_t * z_{t-j}` are its autocovariances. At lag zero `g_0 = 1 - S * k3 + (S^2 / 4) * (k4 - 1)`, `k3` and `k4` the means of `u^3` and `u^4`, the latter 3 on a normal series.
- The lag is Newey and West's 1994 rule for the Bartlett kernel, on `z` itself: `p = floor(4 * (n / 100)^(2/9))`, `s0 = g_0 + 2 * sum_{j=1..p} g_j`, `s1 = 2 * sum_{j=1..p} j * g_j` and `m = min(floor(1.1447 * ((s1 / s0)^2 * n)^(1/3)), n - 1)`, or 0 where `s0` is not above zero. Nothing is prewhitened.
- The variance is the Bartlett sum at that lag floored at its lag-zero value, `V = max(g_0 + 2 * sum_{j=1..m} (1 - j / (m + 1)) * g_j, g_0)`, so autocorrelation can widen an interval and never narrow it.
- The Sharpe's standard error is `sqrt(V / (n - 1))` a bar, and its interval is the per-bar Sharpe `xbar / s'` plus and minus `z_q` of them, `z_q` the normal quantile at `(1 + level) / 2`, both annualised by `sqrt(ppy)`. It names its lag, "delta method, Newey-West lag 9", and where the floor binds says so, "delta method, Newey-West lag 9 floored at lag zero".

**Worked example.** Eight monthly returns, 2%, -1%, 3%, 0%, 1%, -2%, 4% and 1%, with `periods_per_year=12` and no risk-free rate.
- Their mean is 1% and `s'` is 2%, so the Sharpe is 0.5 a bar and `0.5 * sqrt(12) = 1.73` a year. With 8 in the divisor `s` is `sqrt(3.5)` percent, so `S = 1 / sqrt(3.5) = 0.5345`, `k3 = 0` and `k4 = 2`.
- The deviations from the mean are 1, -2, 2, -1, 0, -3, 3 and 0 percent, and for a deviation `d`, `z = (14 * d + 7 - 2 * d^2) / (14 * sqrt(3.5))`, so `14 * sqrt(3.5) * z_t` is 19, -29, 27, -9, 7, -53, 31 and 7. Then `g_0 = 5,880 / 5,488 = 1.0714`, `g_1 = -3,437 / 5,488 = -0.6263` and `g_2 = 1,286 / 5,488 = 0.2343`.
- `p = floor(4 * 0.08^(2/9)) = 2`, `s0 = 1,578 / 5,488` and `s1 = -1,730 / 5,488`, so `m = floor(1.1447 * (1.2019 * 8)^(1/3)) = floor(2.434) = 2`.
- The Bartlett sum `g_0 + 2 * (2/3 * g_1 + 1/3 * g_2) = 0.3926` is below `g_0`, so `V = 1.0714` and the standard error is `sqrt(1.0714 / 7) = 0.3912` a bar.
- The interval is `0.5 +/- 1.96 * 0.3912` a bar, -0.92 to 4.39 a year: "sharpe 1.73, two-sided 95% interval -0.92 to 4.39, delta method, Newey-West lag 2 floored at lag zero, 8 returns".

### The probabilistic Sharpe and the minimum track record

- The probabilistic Sharpe against an annual Sharpe `B` is `Phi((xbar / s' - B / sqrt(ppy)) * sqrt(n - 1) / sqrt(V))`, one-sided, with `V` the variance above.
- The minimum track record at a confidence `c` is `1 + V * (Phi^-1(c) / (xbar / s' - B / sqrt(ppy)))^2` returns, the count at which the same Sharpe and variance would give that probability, and its years are that count over `ppy`. A Sharpe not above the benchmark has none.
- Both carry the variance's lag and whether its floor bound.

**Worked example.** The eight monthly returns above, against zero: `Phi(0.5 * sqrt(7) / sqrt(1.0714)) = Phi(1.278) = 0.899`, and at 95% the track record is `1 + 1.0714 * (1.6449 / 0.5)^2 = 12.6` returns, 1.05 years, where the series has 8, 0.67 years. Against an annual Sharpe of 1, `0.5 - 1 / sqrt(12) = 0.2113` a bar, so the probability is `Phi(0.540) = 0.705` and the track record 65.9 returns, 5.49 years.

**Worked example.** Without autocorrelation `V = g_0`, and both reproduce Bailey and Lopez de Prado (2012). A monthly Sharpe of 0.458 over 24 months with a skew of -2.448 and a kurtosis of 10.164 has `V = 1 + 2.448 * 0.458 + (9.164 / 4) * 0.458^2 = 2.6018` and a probabilistic Sharpe of `Phi(0.458 * sqrt(23) / sqrt(2.6018)) = 0.913`, 0.982 at a normal skew and kurtosis and 0.953 over 36 months (figure 6). A monthly Sharpe of `2 / sqrt(12)` against `1 / sqrt(12)` with a skew of -0.72 and a kurtosis of 5.78 needs 59.895 months at 95% (appendix A.3).

### The long-run Sharpe

`sharpe(x, long_run=True)` is `sqrt(ppy) * mu / sqrt(L)`: `L = c_0 + 2 * sum_{j=1..m} (1 - j / (m + 1)) * c_j` is the Bartlett long-run variance of the returns, `c_j = (1/n) * sum_{t > j} (x_t - mu) * (x_{t-j} - mu)`, at the lag the 1994 rule above takes on `x - mu`. It is not floored, so returns that reverse raise it, and it is above zero wherever the returns vary. A summary prints it beside the Sharpe where the two differ by more than 0.1.

**Worked example.** The eight monthly returns: in squared percent `c_0 = 28 / 8 = 3.5`, `c_1 = -17 / 8 = -2.125` and `c_2 = 7 / 8 = 0.875`, so with `p = 2`, `s0 = 3.5 + 2 * (-2.125 + 0.875) = 1` and `s1 = 2 * (-2.125 + 2 * 0.875) = -0.75`, the lag is `floor(1.1447 * (0.5625 * 8)^(1/3)) = floor(1.890) = 1` and `L = 3.5 - 2.125 = 1.375`. The long-run Sharpe is `sqrt(12) * 1 / sqrt(1.375) = 2.95` a year, beside a Sharpe of 1.73.

### Intervals of the returns

- Total return and net profit, CAGR, volatility, Sortino, value at risk and conditional value at risk take a stationary block bootstrap of the run's returns (Politis and Romano 1994). A resample is `n` returns: its first index is drawn evenly from the `n`, and each next index is the one after the last, wrapping from the last return to the first, with probability `1 - 1/b`, and drawn evenly again otherwise, so its blocks have a mean length `b`. `B` resamples are drawn, 2,000 unless `resamples` gives another count. Value at risk is the loss of the `k`-th lowest return, `k = ceil(n / 20)`, and conditional value at risk the mean loss of the `k` lowest, in percent.
- `b` is the largest of the stationary lengths that Politis and White's rule, as Patton's code computes it (`opt_block_length_REV_dec07`), gives three series: the returns, their squared deviations from their mean, and the indicator of a return at or below the `k`-th lowest, so the dependence in a strategy's squared returns, the stretches its exposure switches on and off, sets the blocks where the returns' own correlogram does not see it. On a series `y` of `N` values, counted from 1:
  - `K = 5`, `m_max = ceil(sqrt(N)) + K`, `b_max = ceil(min(3 * sqrt(N), N / 3))` and the threshold `h = 2 * sqrt(log10(N) / N)`.
  - `rho_j`, for `j` from 1 to `m_max`, is the correlation of `y_t` with `y_{t-j}` over `t` from `m_max + 1` to `N`, each side centred on its own mean. `m_hat` is the first `j` whose `rho_j` to `rho_{j+K-1}` are all below `h` in magnitude, or where no `K` in a row are, the largest `j` whose `rho_j` is above it, and `M = min(2 * m_hat, m_max)`.
  - `R_j`, for `j` from 0 to `M`, is the covariance of `y_t` with `y_{t-j}` over `t` from `M + 1` to `N`, each side centred on its own mean. With the flat-top weights `w(v) = 1` for `v < 1/2` and `2 * (1 - v)` from `1/2` to 1, `G = 2 * sum_{j=1..M} w(j / M) * j * R_j` and `g = R_0 + 2 * sum_{j=1..M} w(j / M) * R_j`, and the length is `abs(G / g)^(2/3) * N^(1/3)`, held to `b_max`.
  - Where it departs from Patton's code, a length is held to at least 1, the shortest mean block a stationary bootstrap has; a series that never varies, holds fewer than `m_max + 3` values or gives a correlation a side that never varies gives 1, and a `g` of zero gives `b_max`.
- Total return and net profit, CAGR, volatility, Sortino and conditional value at risk are studentized (Hall 1988), each read on a pivot `theta` with an influence series `psi`, `n` in every divisor:
  - total return, net profit and CAGR on the growth a bar `g = (1/n) * sum_t ln(1 + r_t)`, `psi_t = ln(1 + r_t) - g`, total return and net profit being `exp(n * g) - 1` and CAGR `exp(ppy * g) - 1`, held to `1e12`;
  - volatility on `ln s'`, `psi_t = ((r_t - m)^2 - v) / (2 * v)`, `m` and `v` the returns' mean and variance, volatility being `sqrt(ppy) * s'`;
  - Sortino on the Sortino a bar `S = xbar / d`, `d^2 = (1/n) * sum_t min(x_t, 0)^2`, `psi_t = (x_t - xbar) / d - (S / (2 * d^2)) * (min(x_t, 0)^2 - d^2)`, annualised by `sqrt(ppy)`;
  - conditional value at risk on the mean `L` of the `k` lowest returns, `psi_t = q + (n / k) * (r_t - q) * [r_t <= q] - L`, `q` the `k`-th lowest, conditional value at risk being `-L`.
- `V` is the Bartlett sum of `psi`'s autocovariances at the lag the 1994 rule takes on `psi`, floored at its lag-zero value, as the Sharpe's is, and the standard error is `e = sqrt(V / (n - 1))`. Each resample gives `t = (theta* - theta) / e*`, `theta*` its pivot and `e*` the same standard error of its own influence series at the sample's lag; a resample whose `e*` is zero or not finite has `t` infinite in the sign of `theta* - theta`, or 0 where the two are equal, and a resample's Sortino with no return below the rate is ranked as the record ranks it. The interval is `theta - t_hi * e` to `theta - t_lo * e`, `t_lo` and `t_hi` the order statistics of the `t` at `(1 - level) / 2` and `(1 + level) / 2`, carried to the statistic, so a volatility's bounds stay above zero and a total return's above -100%.
- Value at risk's interval is percentile, its bounds the order statistics of the resamples' values at the same two levels, each computed as the record computes it. An order statistic at `q` is the `ceil(B * q)`-th lowest of the `B`, the first at least, a product within `1e-9` of a whole number taken as that number, so 95% of 2,000 gives the 50th and the 1,950th where its binary value would give the 51st, never interpolated, so resamples ranked at infinity can make a bound infinite and never NaN. Total return's interval covers `exp(n * E[ln(1 + r)]) - 1`, the median of compounded outcomes over the run's length, CAGR's `exp(ppy * E[ln(1 + r)]) - 1`, and the others the population value of their statistic.
- Maximum drawdown and Calmar take the same order statistics of the same resamples as the middle 95% of resampled histories, which says what histories like the run's do and is not a confidence interval, so it is held to no coverage.

**Worked example.** 1,000 returns whose correlogram never reaches `2 * sqrt(3 / 1,000) = 0.110` in magnitude have `m_hat = 1` and `M = 2`, whose weights at lags 1 and 2 are 1 and 0, so `G = 2 * R_1`, `g = R_0 + 2 * R_1` and the length is `abs(2 * R_1 / (R_0 + 2 * R_1))^(2/3) * 1,000^(1/3)`: at `R_1 = 0.05 * R_0`, `(0.1 / 1.1)^(2/3) * 10 = 2.02`. At the population autocovariances of returns correlated `0.7^j` at lag `j`, `G / g = 2 * 0.7 / (1 - 0.49)` and 200 returns give `2.745^(2/3) * 200^(1/3) = 11.47`, as Patton, Politis and White (2009) print in their table 1.

### Intervals of the round trips

The win rate, the profit factor and the average trade take a bootstrap of the closed round trips' pnl: `B` resamples, 2,000 unless `resamples` gives another count, of `k` trips drawn evenly with replacement from the `k` closed, each statistic computed on each as the record computes it.
- The win rate's interval is percentile, as above.
- The profit factor's is bias-corrected and accelerated (Efron 1987): `z0 = Phi^-1(f)`, `f` the share of resamples below the estimate with a tie counting half, and `a = sum_i (P - P_i)^3 / (6 * (sum_i (P - P_i)^2)^(3/2))`, `P_i` the profit factor without trip `i` and `P` the mean of those. Its bounds are the order statistics, ranked as above and the first at least, at `Phi(z0 + (z0 + z) / (1 - a * (z0 + z)))` for `z` the normal quantiles at `(1 - level) / 2` and `(1 + level) / 2`. Resamples that all fall on one side of the estimate leave `z0` without a value, and the interval is refused, asking for more of them.
- The average trade's is studentized: each resample gives `t = (m* - m) / (s* / sqrt(k))`, `m` and `m*` the mean pnl of the trips and of the resample and `s` and `s*` their deviations with `k - 1` in the divisor, and the interval is `m - t_hi * s / sqrt(k)` to `m - t_lo * s / sqrt(k)`, `t_lo` and `t_hi` the order statistics of the `t` at `(1 - level) / 2` and `(1 + level) / 2`, in percent of the opening balance. A resample whose trips all made the same has `t` infinite in the sign of `m* - m`, or 0 where the two are equal.

### Samples with no interval

- Fewer than 4 returns, or returns that never vary, give no return statistic and no Sharpe an interval. Returns whose Sharpe's influence is zero on every bar, which its quadratic allows only for two values in one proportion, leave it no standard error, and it has no interval, probability or track record.
- A bootstrap of the returns needs 500 of them, holding at least 100 of its mean blocks. Below 500 the coverage cells' heavy-tailed and autocorrelated returns cover too little, and the bootstrap's variance falls short by about one over the square root of its count of blocks (Politis and White 2004), so returns that switch in and out of the market in stretches too long for their length get none. A run short enough to sit in one stretch reads as independent, and its interval is that stretch's.
- A return of -100% or worse gives total return, net profit and CAGR none, since the growth they are read on has no value past it.
- Conditional value at risk needs a tail that varies: where the `k` lowest returns are all equal, as for a run that never lost on a bar, it has none, and nor has any other studentized statistic whose standard error is zero on the sample.
- An interval whose resamples put both its bounds on one value has none, as value at risk's where its `k`-th lowest sits among many flat bars, a win rate's at a narrow level, or any interval at a level too narrow for its resamples. A range of resampled histories is kept as they give it, though they meet at one value, as for returns that never lose, whose every resampled drawdown is zero.
- Fewer than 60 closed round trips give no trade statistic one: at 60 every trade statistic's interval holds to the coverage cells' band, and at 50 the average trade of a strategy that wins often and small sits on its edge. The win rate needs a trip that won and one that did not, the profit factor one that won and two that lost, since its acceleration leaves out each trip in turn, and the average trade trips that did not all make the same.
- An estimate ranked at infinity has no interval: no resample of the sample can measure a risk the sample lacks, as for a profit factor with no losing trip, a Sortino with no return below the rate and a Calmar with no drawdown.
- A Result's statistics hold None for such an interval. Asked for directly, it is refused, naming why and what would give one: "returns: 3 of them; an interval needs at least 4", or "returns: 600 in blocks of 29.7, 20.2 of them; a bootstrap of returns needs at least 100 blocks".

### Two Sharpes on the same bars

`sharpe_difference(a, b)` compares two Results on the same rows, or two plain series of one length, or a Result alone with its buy-and-hold baseline on the same rows at the Result's own risk-free rate, which a run at `account` does not have. `D` is the difference of the two per-bar Sharpes, and both methods are reported, each naming itself.
- The delta method: the influence series `z_a - z_b`, each as for one Sharpe, its variance `V` the Bartlett sum floored at its lag-zero value at the lag the 1994 rule takes on it, `se = sqrt(V / (n - 1))`, the p-value `2 * Phi(-abs(D) / se)` and the interval `D +/- z_q * se`, annualised.
- Ledoit and Wolf's (2008) studentized circular block bootstrap: a resample joins `ceil(n / l)` blocks of `l` rows, each starting at a row drawn evenly and wrapping past the last, the same rows of both series, cut to `n`. `l` is Patton's circular length on `z_a - z_b`, the ceiling of `(3/2)^(1/3)` times the stationary length before it is held, held to `b_max` and at least 1. Each resample gives `D*` and its natural standard error `s* = sqrt(V* / (n - 1))`, `V* = (1/q) * sum_{i=1..q} (sum of z* over block i)^2 / l` over its `q = floor(n / l)` whole blocks, `z*` its own influence series, and `d* = abs(D* - D) / s*`, infinite for a resample whose returns never vary or whose `s*` is zero. The p-value is `(#{d* >= abs(D) / se} + 1) / (B + 1)` over the `B` resamples and the interval `D +/- d_level * se`, `d_level` the order statistic of the `d*` at `level`.
- Two series whose influence series are the same on every bar, a series beside itself or beside a positive multiple of it at no risk-free rate, or two whose influence is zero on every bar, leave the difference no standard error, and are refused. A variance below `1e-20` counts as zero, here and for one Sharpe: the influence is standardised, so its rounding stays below that while the Sharpe is under about a thousand a bar.
- Both depart from that paper in two places: Newey and West's Bartlett kernel at the 1994 lag stands for its prewhitened quadratic-spectral kernel, and Politis and White's block length for its calibration.

**Worked example.** At lag zero the delta method is Memmel's (2003) correction of Jobson and Korkie. Two normal series with per-bar Sharpes of 0.10 and 0.05 correlated 0.5 have `E[(z_a - z_b)^2] = 2 - 2 * 0.5 + (0.01 + 0.0025 - 2 * 0.005 * 0.25) / 2 = 1.005`, so over 1,000 bars `se = sqrt(1.005 / 999) = 0.0317` and a difference of 0.05 a bar has a p-value of `2 * Phi(-1.576) = 0.115`. If 113 of 2,000 resamples have `d*` at or above `abs(D) / se`, the bootstrap's p-value is `114 / 2,001 = 0.057`.

### Alpha and beta

`alpha_beta(result, benchmark)` regresses the run's excess returns on the benchmark's with an intercept, `x_t = alpha + beta * y_t + e_t`, by least squares over the run's `n` returns.
- The covariance is Newey and West's, `(X'X)^-1 * W * (X'X)^-1` for the rows `X_t = (1, y_t)` and `W = sum_t e_t^2 * X_t X_t' + sum_{j=1..m} (1 - j / (m + 1)) * sum_{t > j} e_t * e_{t-j} * (X_t X_{t-j}' + X_{t-j} X_t')`, with no small-sample factor. `m` is the given `lag`, or the 1994 rule above on the slope's score `y_t * e_t`.
- Alpha is annualised by `ppy` and beta is per bar, each with its standard error and its interval `+/- z_q` of them, alpha with its t statistic. The tracking error is the deviation of `x_t - y_t`, `n - 1` in its divisor, times `sqrt(ppy)`, and the information ratio the mean of `x_t - y_t` over that deviation, times `sqrt(ppy)`.
- The benchmark is the run's buy-and-hold baseline unless one is given: a Result on the same rows, an array shaped as the run's returns, its row 0 ignored, or a pandas Series indexed by the run's row times exactly.

**Worked example.** A benchmark returns -2%, -1%, 0%, 1% and 2% over five months and a strategy -1%, -0.5%, 0.5%, 1.5% and 2%, with no risk-free rate.
- The residuals of `0.5% + 0.8 * y` are 0.1%, -0.2%, 0%, 0.2% and -0.1%, which sum to zero and to zero against the benchmark, so alpha is 0.5% a month, 6% a year, and beta is 0.8. In percent `X'X = [[5, 0], [0, 10]]`.
- The slope's score is -0.2, 0.2, 0, 0.2 and -0.2, whose `s1` is zero, so the lag is 0, `W = [[0.1, 0], [0, 0.16]]`, and the standard errors are `sqrt(0.1) / 5 = 0.063%` a month for alpha and `sqrt(0.16) / 10 = 0.04` for beta.
- With `lag=1`, the two adjacent residual products of -0.02 take `[[0.04, 0], [0, 0.08]]` from `W`, and the standard errors are 0.049% and 0.028.
- The strategy beats the benchmark by 1%, 0.5%, 0.5%, 0.5% and 0%, a mean of 0.5% and a deviation of 0.354%, so the tracking error is `0.354 * sqrt(12) = 1.22%` and the information ratio `0.5 / 0.354 * sqrt(12) = 4.90`.

### The order of trades

`trade_order(result, n=2000)` permutes the closed round trips `n` times, every instrument's in one sequence, which in the run is the order they closed.
- Trip `i` returns `pnl_i / E_i`, `E_i` the account's equity at the close of the row before its opening fill, or the opening equity for a trip opened on the first row, and an order compounds the returns from the opening equity, staying at zero once it reaches it: the model of a policy sized on its equity, so the order is all that moves.
- The drawdown is the largest fall from the running peak at trade closes, the opening equity included, as a share of the peak and at most a total loss, and the losing run the longest run of trips whose pnl is below zero.
- It reports the run's own drawdown and losing run, the middle 95% of the permutations' for each, the share of permutations whose drawdown is deeper by more than a part in a trillion, so that orders apart only by rounding count as level, and `n`, and says the drawdown is at trade closes and leaves out what the account marked inside a trade. Fewer than 2 closed round trips cannot be permuted.

**Worked example.** Three round trips return 20%, -10% and -10% on the equity they opened on, from 10,000. Every order ends at 9,720. In the run's order the drawdown is 19%, from 12,000 to 9,720, with a losing run of 2; the four orders of the six that put the two losses together draw down 19%, and the two that part them 10%, with losing runs of 1. So the middle 95% of the permutations' drawdowns is 10% to 19% and of their losing runs 1 to 2, and none draws down deeper than the run.

### Summaries

- `Result.summary()` prints the run's header and span, its return, its Sharpe with the interval and the lag or why it has none, the long-run Sharpe beside it where they differ by more than 0.1, its maximum drawdown as a loss with the longest time under water in days, from a row at the running peak to the first row back at or above it or to the last row, its round trips with the win rate and the profit factor, its costs and its assumptions. A combined book names each account and says the accounts are margined apart and no capital moves between them, and a run recorded below the full level names its level.
- A search's `summary(null=None, plateau=None)` prints its header, its in-sample span, the best trial's parameters and score, and its holdout. `null` is a random search of the same policy over the same space on the same venue and in-sample bars, scored by the same objective, direction, `min_trades` and risk-free rate, and is refused naming what differs; its line is the mean score of its complete trials and their count, "null mean sharpe 0.38 over 500 trials", after the deflated Sharpe's line, which reads it (Across a search). Before the holdout a search prints its overfitting probability's text, or why it has none: no returns kept, an objective the halvings cannot read, fewer than 2 complete trials, or fewer returns than its 16 blocks. With `plateau`, a level in the objective's units as the trials table holds them, the plateau's text follows it ([The plateau](#the-plateau)). An adaptive search's lines say "descriptive".
- A validation's `summary()` prints its header and its paths, the embargo, each path's score, the median of the complete paths' scores, and the interval of the median path's objective on its returns, or the range of its resampled histories for drawdown and Calmar, the median path being the lower of the two middle ones for an even count, with "the paths share bars"; where the returns were not kept, or the objective reads round trips, which a path's returns do not hold, it says so and gives none. A model's then prints its pooled forecast, "forecast rank IC 0.021, t = 2.4 (Newey-West, lag 25); R squared 0.3%", or why it has none. Then its costs as a share of the gross edge, the net profit plus the costs, each summed over the complete paths, or "costs N, with no gross edge", the label "the paths are stability evidence; the verdict comes from the holdout", and the holdout.
- The holdout reads as its state:
  - none held out: "holdout none: the search held every bar";
  - held out and not yet run: "holdout not yet run; evaluate_holdout() spends a look", the line before naming what is held out, "holdout the last 30% (9,641 bars)";
  - run: "holdout sharpe 0.52, 95% interval -1.35 to 2.39, look 1 of 1", and after `k` looks the interval at the level `1 - 0.05 / k` beside it;
  - a look whose run raised: "holdout read 2 times; its run raised, so no result is kept";
  - looks with no result kept for this winner, whose runs the Study did not see: another Study's of its search where it was saved or kept, ones from before a resume changed its winner, and a loaded Study's taken after it was saved: "holdout read 1 time before, with no result kept for this winner; evaluate_holdout() spends another look".

### Consistency and decay

- `consistency(stretches=None)` scores a search's best trial on `stretches` stretches of its run over the in-sample bars, a whole number from 2 and 4 unless given: its rows after the first divided as `PurgedKFold` divides bars, each scored by the objective as a validation's window is, opening at the equity of the close before it. A validation's stretches are its windows, and `stretches` given to one is refused.
- A stretch gains where its score is past the objective's break-even in the search's direction: 0 for `sharpe`, `sortino`, `calmar`, `total_return_pct`, `net_profit_pct`, `cagr_pct` and `avg_trade_pct`, and 1 for `profit_factor`. The other objectives have no break-even, so their scores are given, the share is None and the text says why.
- `decay()` is the mean gap between what a fit scored and what its bars then earned, signed so that a positive decay is worse out of sample in either direction. A validation's gaps are each window's fit's in-sample score against the window's score. A search's one gap is its best trial's in-sample score against its holdout's, which needs a look whose run kept its result. With nothing held out it is refused naming the holdout to search with; with no look, a look whose run raised, or looks counted with no result kept for this winner, naming what was or was not kept and `evaluate_holdout()`, which spends a look; and on a loaded Study with no holdout run, naming the search into the directory it was saved in, which runs only what is not there.
- An objective that grows with its span, `total_return_pct`, `net_profit_pct`, `max_drawdown_pct`, the counts or `funding_paid`, has no decay, since spans of different lengths would make the gap measure their lengths, and neither has a model's validation, whose fits score no bars of their own.

**Worked example.** A search maximising the Sharpe whose best trial ran over 1,001 in-sample rows scores 0.81, 1.22, -0.30 and 0.94 on rows 1 to 250, 251 to 500, 501 to 750 and 751 to 1,000: "3 of 4 stretches scored above 0". A validation whose three windows' fits scored 1.8, 1.2 and 2.0 in sample and whose windows scored 0.6, 1.4 and -0.2 has gaps of 1.2, -0.2 and 2.2, a decay of 1.07.

<br>
<br>


## Across a search

`probatum.stats` also judges a search across all of its trials: how far the winner's Sharpe clears what the best of that many trials reaches by luck, how often the in-sample winner falls below the median out of sample, whether any trial beats a benchmark once every trial is counted, how long a backtest that many trials need, and whether the winner stands among neighbours that score as it does. Each reads a Study of trials or a matrix of trials' returns, the plateau a Study alone, and states how many trials it read.

### What a statistic of trials reads

- A Study's trials are the rows of its `returns`, each trial's `n = T - 1` returns after the in-sample bars' first row, as a Result's are, at the Study's annualisation and risk-free rate, so a Study refuses `periods_per_year` and `risk_free`. A plain matrix, one row a trial's returns and every row on the same bars, needs `periods_per_year` and takes `risk_free` as an annual rate, 0 unless given; a NaN or an infinity in it is refused, naming its trial and position.
- The deflated Sharpe counts every trial the search asked as a look, complete, floored and failed. The effective number of trials and the Reality Check read every trial that ran, complete and floored, and the overfitting probability the complete trials, the ones that could win. A trial whose returns never vary is left out of the clustering, and the count left out is reported.
- A Study kept with `keep="none"` holds no returns, and what reads them is refused, naming `keep="all"`. The deflated Sharpe against every trial reads only the null's statistics and the winner's run, so it stands, its effective part None, and the plateau reads the trials table alone.
- A Study of paths is refused by each: its rows are no competing configurations.
- An adaptive search's trials were chosen on the bars they are judged on, so its deflated Sharpe, overfitting probability, Reality Check and SPA are computed and labelled descriptive. A random or grid search chooses its trials without reading a score, so those are tests.
- The clustering's starts draw from the stream `stats onc`, seeded by the trials' returns, and the Reality Check's resamples from `stats reality check`, seeded by the trials' returns and the benchmark's, so each is the same on every read and in every process, and `seed=` stands in for the digest.

### The deflated Sharpe

- The expected maximum of `N` independent standard normal draws is approximated as `f(N) = (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 - 1/(N * e))`, `gamma = 0.5772156649015329` the Euler-Mascheroni constant and `e` Euler's number (Bailey and Lopez de Prado 2014, the false strategy theorem). It is within 0.045 of the exact expected maximum, above it from 3 draws on and below it at 2.
- `deflation_threshold(spread, trials)` is `spread * f(trials)`, the annual Sharpe the best of `trials` independent trials reaches by luck when their annual Sharpes spread `spread` about zero. It needs 2 trials and a spread of at least 0.
- `deflated_sharpe(study, null)` reads a search that ranked on the Sharpe, maximised, and a null as a summary's is (Summaries). `N` is the number of trials the search asked. The spread `sigma` is the sample deviation, with its count less one in the divisor, of the finite annual Sharpes of every trial of the null that ran, complete and floored, so an activity floor keeps no trial out of it. The threshold is `sigma * f(N)`, and the value is the winner's probabilistic Sharpe against it, on the variance of The Sharpe's variance, read from the winner's run on the in-sample bars. A null that asked fewer trials than the search warns, since its spread then rests on fewer draws than the trials it corrects for.
- At the effective number of trials the search's trials are clustered into `K` (The effective number of trials), and the threshold is `sigma_K * f(K)`, `sigma_K` the sample deviation of the K clusters' annual Sharpes, or 0 at `K = 1`, the expected maximum of one draw; the value is the probabilistic Sharpe against it of the winner's cluster, the series its members combine into, on the variance of The Sharpe's variance, or of the winner's run where its returns never vary and it is in no cluster.
- Its text: "deflated sharpe 0.64 against all 500 trials; 0.79 for the winner's cluster against 23 effective trials".

**Worked example.** Bailey and Lopez de Prado's: 100 trials whose Sharpes spread with a variance of 1/2 a year, and a winner whose 1,250 daily returns at 250 a year have an annual Sharpe of 2.5, a skew of -3 and a kurtosis of 10, at lag zero. `f(100) = 2.5306`, so the threshold is `sqrt(0.5) * 2.5306 = 1.7894` a year, `0.1132` a bar. The winner's Sharpe of `0.1581` a bar has `V = 1 + 3 * 0.1581 + (9 / 4) * 0.1581^2 = 1.5306` and a deflated Sharpe of `Phi((0.1581 - 0.1132) * sqrt(1,249) / sqrt(1.5306)) = Phi(1.2838) = 0.9004`. After 46 trials it would be 0.9505; with normal returns, 0.9505 after 88 trials and 0.9498 after 89.

### The effective number of trials

- The `R` trials that ran and whose returns vary are clustered where `R` is at least 3, and with fewer each is a cluster of its own.
- `rho` is the Pearson correlation of their returns and `D = sqrt((1 - rho) / 2)`, and each trial is the point its row of `D` makes in `R` dimensions, so two trials are `||D_i - D_j||` apart, the distance of their distances to every trial (Lopez de Prado and Lewis 2019).
- A clustering's silhouettes are `s_i = (b_i - a_i) / max(a_i, b_i)`, `a_i` the mean distance from trial `i` to the other members of its cluster and `b_i` its least mean distance to another cluster's members, 0 for a trial alone in its cluster. Its quality is `q = mean(s) / sd(s)`, `R` in the deviation's divisor, infinite where the deviation is 0 and the mean above 0, and 0 where both are 0.
- The base clustering. For each of 10 starts, one greedy k-means++ sequence of `k_max = min(R - 1, 100)` trials is drawn from the stream: the first evenly, and each next the best of `2 + floor(ln k_max)` trials drawn with probabilities in proportion to their squared distance from the sequence so far, an even draw `u` taking the first trial whose running sum of those distances is above `u` times their total, and each drawn evenly where every distance is 0, best by the sum of every trial's squared distance to its nearest, the first of equal ones. For each `k` from 2 to `k_max`, Lloyd's iterations run from the sequence's first `k` trials as centres: each trial joins its nearest centre, the lowest-numbered on ties; each cluster left empty, in order, takes the trial farthest from its centre that has not moved, and one emptied by that keeps its centre; each centre moves to its members' mean; until no trial moves, or 300 times. The base is the clustering of the highest quality, the first in start and then `k` order on ties.
- The refinement. Each cluster of more than one trial has a quality, its members' silhouettes' mean over their deviation. Where more than 2 clusters fall below the mean of those qualities, their trials are clustered again, base and refinement, on their own correlations, and the new clusters replace them where the new clusters' mean quality, on the silhouettes of the whole new clustering, is above the mean quality of the clusters they replace.
- `K` is the number of clusters. Each cluster's series combines its members' returns at minimum variance under one correlation, weights in proportion to `1 / d_i^2 - rbar * (sum_j 1 / d_j) / ((1 + (m - 1) * rbar) * d_i)` and summing to 1, `d_i` the members' deviations, `m` their count and `rbar` their mean pairwise correlation, or `1 / d_i^2`, inverse variance, where `rbar` is not within `(-1 / (m - 1), 1)`; a cluster of one trial is that trial's returns. `sigma_K` is the sample deviation of the K series' annual Sharpes.
- Its text: "23 effective trials of 500, quality 2.72, their clusters' sharpes spread 0.41 a year".

**Worked example.** Lopez de Prado's 6,385 bond backtests fall into four clusters whose series have annual Sharpes of 1.5733, 1.4907, 2.0275 and 1.0158 (2019, exhibit 6). Their variance is `0.5153 / 3 = 0.1718` a year; at 261.12 bars a year that is a deviation of `sqrt(0.1718 / 261.12) = 0.02565` a bar, and the threshold at four trials is `0.02565 * f(4) = 0.02565 * 1.0521 = 0.0270` a bar.

### The probability of backtest overfitting

- `overfitting(trials)` ranks on the search's objective, in its direction; a matrix ranks on `objective`, `"sharpe"` unless given, maximised, or volatility minimised. The objective is read from the returns alone and does not depend on their order: the Sharpe, Sortino, total return, net profit, CAGR or volatility. Drawdown and Calmar depend on the order of the returns, and value at risk and the round trips' statistics are not read from the halves, so each is refused.
- The returns are cut into 16 contiguous blocks of `floor(n / 16)`, the oldest `n mod 16` left out, so a run of fewer than 16 returns is refused.
- Every one of the `C(16, 8) = 12,870` ways of choosing 8 blocks is a halving: the 8 chosen, joined in time order, are in sample, and the other 8 out of sample. Each half's statistic is computed as the record computes it, a zero denominator ranking at infinity for a positive reward.
- In each halving the in-sample winner is the trial with the best statistic; its out-of-sample rank is its mid-rank among the trials, `r = #{worse} + (#{equal} + 1) / 2` with itself among the equal, `omega = r / (N + 1)` and its logit `lambda = ln(omega / (1 - omega))`. Where `k` trials tie as the winner, each counts with weight `1 / k`.
- The probability of backtest overfitting is the share of halvings whose winner's logit is below 0, its out-of-sample rank below the median, a winner exactly at the median counting as not overfit (Bailey, Borwein, Lopez de Prado and Zhu 2017, definition 2.2). The probability of loss is the share whose winner's out-of-sample statistic is below 0, for an objective with a sign.
- It needs 2 complete trials. Its text: "overfitting 0.41 probability the in-sample best ranks below the median on the held-back blocks (16 blocks, 500 trials); it loses on 23% of them".

**Worked example.** With 4 blocks in place of 16, so that its 6 halvings can be listed: four trials of 8 returns, each block 2 equal returns, (0.4, 0.1, -0.2, 0.3), (-0.1, 0.5, 0.2, -0.3), (0.2, -0.4, 0.6, 0.1) and (0.35, 0.45, 0.05, 0.15) a block, ranked by their mean. In the halving AB against CD the in-sample means are 0.25, 0.2, -0.1 and 0.4, so trial 4 wins; out of sample its 0.1 ranks third of 0.05, -0.05, 0.35 and 0.1, `omega = 3 / 5`, `lambda = ln 1.5 = 0.41`, not overfit. In AC against BD trial 3 wins and ranks last, `lambda = ln(1 / 4) = -1.39`. Four of the six halvings are overfit and four winners lose out of sample: both probabilities are 4/6, 0.67.

### The Reality Check and SPA

- `reality_check(trials, benchmark=None)` asks whether any trial beats `benchmark`: cash, earning the risk-free rate a bar, unless `"buy_and_hold"`, a Study's buying and holding of its instruments on the in-sample bars as a Result's baseline, or a series of returns on the same bars, a value a row of a Study's in-sample bars, its first ignored, as its returns hold them, or a value a column of a matrix.
- Each trial that ran gives `f_k,t = ln(1 + r_k,t) - ln(1 + b_t)`, its log growth over the benchmark's a bar (Sullivan, Timmermann and White 1999), and `fbar_k` its mean. A trial with a return at or below -100% has no log growth and is left out, and the count reported; a benchmark with one is refused, naming its position.
- White's statistic is `V = max_k sqrt(n) * fbar_k`.
- `B` resamples, 1,000 unless given, are drawn once and shared by every trial: each a stationary bootstrap of the `n` bars, its first row drawn evenly and each next the row after the last, wrapping past the last, with probability `1 - q`, or drawn evenly again (Politis and Romano 1994). The mean block `1 / q` is the median of Politis and White's stationary lengths on the `f` of the trials whose `f` varies, each as Patton's code computes it and held to `[1, ceil(min(3 * sqrt(n), n / 3))]` (Intervals of the returns), and 1 where none varies.
- White's p-value is the share of resamples with `max_k sqrt(n) * (fbar*_k - fbar_k) > V` (White 2000).
- Hansen's `omega_k^2 = g_0 + 2 * sum_{i=1..n-1} kappa(n, i) * g_i`, `g_i` the autocovariances of `f_k` with `n` in the divisor and `kappa(n, i) = (1 - i/n) * (1 - q)^i + (i/n) * (1 - q)^(n - i)`, is the stationary bootstrap's own variance of `sqrt(n)` times a resample's mean. Over the trials whose `f` varies, `t_k = sqrt(n) * fbar_k / omega_k` and `T = max(0, max_k t_k)`. A trial whose `f` never varies beats the benchmark on every bar where its constant is above 0, and SPA's p-value is then 0; otherwise it is left out of SPA's maximum.
- Each resample recentres each trial's mean by `c_k`, Hansen's consistent choice: `fbar_k` where `t_k >= -sqrt(2 * ln(ln n))`, so a trial far below the benchmark keeps its mean, and 0 otherwise. It studentizes each by `omega*_k`, the same variance on the resampled series, or `omega_k` where that series does not vary: `T* = max(0, max_k sqrt(n) * (fbar*_k - c_k) / omega*_k)`. SPA's p-value is the share of resamples with `T* > T`, and 1 where `T = 0` (Hansen 2005, but for the variance recomputed in each resample, where Hansen fixes it at the sample's).
- It needs 3 bars. Its text: "reality check p 0.21 and SPA p 0.18 that no trial beats cash, 1,000 stationary resamples in blocks of 12.3, 500 trials".

**Worked example.** Four trials over 4 bars whose relative series are (1.875, -1.125, 2.875, -0.125), (1, 1, -2, 2), (-3, -2, -4, -3) and (1, -1, 1, -2), with `q = 0.5` and five resamples of rows (2, 3, 4, 1), (1, 1, 2, 3), (4, 1, 2, 2), (3, 3, 3, 4) and (1, 2, 1, 2). The means are 0.875, 0.5, -3 and -0.25, so `V = 2 * 0.875 = 1.75`; the resamples' maxima of `2 * (fbar* - fbar)` are 0, 1.5, 1.5, 2.5 and 1, one above `V`, and White's p-value is 0.2. The variances are `omega^2 = 75/64, 11/8, 19/64 and 199/256`, so `t = 1.6166, 0.8528, -11.0120 and -0.5671` and `T = 1.6166`; the third trial keeps its negative mean, since `-11.01` is below `-sqrt(2 * ln(ln 4)) = -0.8083`, and with each resample studentized on its own variance three of the five maxima pass `T`: SPA's p-value is 0.6.

### The minimum backtest length

- `min_backtest_length(trials, sharpe)` is `(f(N) / S)^2` years for `N` trials and an annual Sharpe `S`, the backtest over which the expected best of `N` independent trials with no skill falls to `S` (Bailey, Borwein, Lopez de Prado and Zhu 2014). It needs 2 trials and a Sharpe above 0. A Study's is its trials' count at its winner's in-sample Sharpe, beside the years its in-sample bars span.
- Its text: "minimum backtest length 3.1 years for the best of 500 trials to fall to a sharpe of 1.41; this series has 2.6".

**Worked example.** 45 trials need `f(45)^2 = 2.2356^2 = 4.998` years to bring their expected best down to a Sharpe of 1, and 46 need `2.2442^2 = 5.036`, so five years of data support no more than 45 configurations (2014, figure 2).

### The plateau

- `plateau(study, level)` reads a search's trials around its winner. An `Int`'s or a `Float`'s value is put on `[0, 1]` across its range on the scale it was drawn on, `(x - low) / (high - low)`, or `ln(x / low) / ln(high / low)` on a log scale, and is 0 where `low` is `high`; a `Categorical`'s is 0 from the winner's where a trial's choice is the winner's, NaN being one choice, and 1 where it is not; and a trial's distance from the winner is the Euclidean distance over its parameters.
- At each radius `r = 0.05 * k`, `k` from 1 to 10, the trials within `r` are those at a distance of at most `r`, a configuration the search ran more than once counted once, by the first trial that ran it, and the winner's own not at all, since a configuration asked for again adds no neighbour; and the share is how many of them completed with a score past `level` in the search's direction, above it maximising and below it minimising, over how many there are, NaN where there are none. A floored or a failed trial within `r` counts against the share, as a configuration beside the winner that scored nothing; a trial the workers' breaking failed never ran, and is no neighbour.
- `level` is a finite number in the objective's units as the trials table holds them.
- Its text gives the share at 0.1 and then at every radius, "22% of the 40 trials within distance 0.1 of the best, in normalised parameters, score above 1; from 0.05 to 0.5: 30%, 22%, 25%, 21%, 18%, 15%, 14%, 12%, 11%, 10%", a loss printed as the negative return it is with its side turned, and "none" for a radius no trial lies within, which a search's summary prints after "plateau".

**Worked example.** A search maximising the Sharpe over `fast` in `Int(5, 45)` and `slow` in `Int(100, 300)`, whose winner is fast 25 and slow 200. A trial at 27 and 210 lies `sqrt(0.05^2 + 0.05^2) = 0.0707` from it, one at 28 and 200 at 0.075, and one at 35 and 240 at `sqrt(0.25^2 + 0.2^2) = 0.32`. Scoring 1.2, 0.4 and 1.1 against a level of 1.0, they leave no trial within 0.05, 1 of 2 past the level within 0.1, and 2 of 3 within 0.35.

<br>
<br>


## Capacity and costs

`capacity`, `cost_curve` and `breakeven_bps` run one policy again where only its balance or its costs differ. Each run is the `backtest` of the policy on the data at the venue with the settings given, `sizing`, `stop`, `take_profit`, `tolerance` and `risk_free`, or an agent's `spec`, `features`, `decode` and `instruments`, at the full recording level, a Strategy on a copy of its own as `backtest` copies one and an agent with no run without costs beside it, so the runs differ in nothing else. An agent's balances and costs are those of the slots it trades, and so is what each run measures.

### What a run of them reads

- A run's traded notional is its fills' notionals at their prices summed, `abs(q) * p`, or `abs(u) / p` on an inverse contract ([Inverse contracts](#inverse-contracts)), in the asset the account settles in.
- Its round-trip cost is `2 * 10,000 * (fees + spread + slippage + impact) / traded` basis points, the costs as the run totals them ([Costs](#costs)), a round trip trading its notional once each way; NaN where nothing filled.
- Its turnover is a year of its traded notional over its mean equity, `traded / mean(E) * ppy / n`, with `E` the equity at each of its rows, `n` its returns and `ppy` its periods a year ([Annualisation](#annualisation)), NaN where it has no annualisation or its mean equity is not above zero.
- The orders the liquidity budget held are those with an event naming it, a clamp it set, a fill of nothing it allowed or a `fok` it refused ([The fill log and the order log](#the-fill-log-and-the-order-log)), as a share of the orders placed, 0 where none was.

### Capacity

- `capacity(policy, data, venue, balances)` runs at each balance of `balances` in turn, the account opening with it as `Venue.replace(balance=...)` opens one, and gives each balance's return, Sharpe, impact paid, share of orders the liquidity budget held, round-trip cost and turnover.
- A weight is a share of equity, so its positions grow with the balance; a quantity does not, and `sizing="quantity"` is refused, as is, before any run, a balance the account cannot open with: one wallet's where the data trades spot and perpetuals, which use both, or two for an inverse contract, whose account holds its coin in one.

**Worked example.** A weight of 1.0 on a linear perpetual decided at a close of 60,000, its next bar opening there with a trailing volume of 5 BTC under `budget(0.1)`. At a balance of 30,000 the order of 0.5 BTC fills whole and no order is held. At a balance of 60,000 the order of 1 BTC fills 0.5 BTC, a clamp naming the liquidity budget, so the one order placed is one it held, and half the position wanted is bought.

### Cost sweeps

- `cost_curve(policy, data, venue, scales=(0, 0.5, 1, 2, 4))` runs at each scale in turn with every cost the venue charges for trading multiplied by it: each fee rate above 0, maker and taker, of the venue, its kinds and its instruments; the spread, a fixed one's basis points or the estimate ([The spread estimate](#the-spread-estimate)); the slippage; and the impact coefficient. A fee rate below 0 is a rebate the venue pays and is not moved, nor are the liquidity budget, funding and a liquidation fee the venue sets, being the market's limit and rates rather than charges for trading; where the venue sets none, the liquidation fee is the taker rate ([The trigger, close and clip points](#the-trigger-close-and-clip-points)) and moves as it does. It gives each scale's round-trip cost, return, Sharpe, round trips, fees, spread, slippage and impact paid, and turnover.
- A scale is a number from 0. Where the venue charges none of the four on the data's instruments the sweep is refused, since no scale of it moves a cost.
- `breakeven_bps(policy, data, venue, ceiling=100, precision=0.05)` is the round-trip cost at the scale where the return reaches zero, `ceiling` a scale above 0 and `precision` basis points above 0:
  - None where the run at scale 0 returns nothing above zero;
  - where the run at `ceiling` still returns above zero, its round-trip cost, warning that the break-even lies past it;
  - and otherwise the mean of the round-trip costs of two runs that bracket a crossing, the lower scale's return above zero and the upper's not, the bracket halved from 0 and `ceiling` until those costs are within `precision` of each other, no scale lies between its ends or 60 halvings have run, warning where the costs are still more than `precision` apart, as where a cost jumps at the crossing; or the lower run's cost where the upper filled nothing.
- The costs change which fills happen and how large, so the return need not fall steadily as the scale grows, and the bisection finds a crossing rather than the crossing.

**Worked example.** The round trip of [Costs](#costs) pays, at scale 1, fees of 6.019997, spread 0.602 and slippage 1.204 on a traded notional of `6,000.90 + 6,039.094 = 12,039.994`, a round-trip cost of `2 * 10,000 * 7.825997 / 12,039.994 = 13.0` basis points: 5 of fee, 0.5 of half the spread and 1 of slippage each way. At scale 2 it fills at 60,018.00 and 60,381.88 and its costs come to 15.651988, 26.0 basis points. Its price PnL of 40.00, less the funding of 0.605 that no scale moves, leaves 39.395 for the costs to take, so its return reaches zero at a scale of about 5.03, where it pays 65.4 basis points a round trip.

<br>
<br>


## The environment

`env(data, venue, window=64, ...)` steps `num_envs` environments together, each an account at the venue on the Data's bars. A step takes each environment's action at the close of its bar `t`, turns it into that close's decision, and returns the observation and the reward at the close of `t + 1`; the decision fills as a policy's would from the open of `t + 1`, so an episode is the run of [The step](#the-step) over its bars, and every fill, fee, funding charge, cap and liquidation in it is worked as in a backtest at the venue. Each environment is worked apart from the others, so a step gives the same results on any number of `threads`.

### Episodes

- The features are a `(T, N, F)` array or a function of a Data that returns one, read as `Data.features` reads them, so a NaN on a printed bar past an instrument's warm-up is refused unless `fill` is given, and a finite value past what float32 holds is refused; without them each instrument's features are its candles. An instrument's features start at `e`, the first bar it printed on with every feature finite, its first printed bar for the candles. It has entered at bar `t` once the window of `W` bars ending at `t` starts at or after `e`, and it is shown at `t` where it has entered, printed at `t` and is not delisted.
- An episode starts at a bar `s` drawn uniformly among the bars from `W - 1` to `T - 2` at which at least one instrument is shown, or `sample` of them with `sample=`, as a fresh account with the venue's balance, observed at the close of `s`. It is truncated at the last bar, `T - 1`, or after `episode_length` steps.
- `sample=k` gives each environment `k` slots, the instruments it trades in an episode, drawn without replacement among those shown at `s` and kept in the Data's order; without it the slots are every instrument. The observation and the actions are over the slots. A universe of spot and perpetual instruments together is not sampled, since its slots would change kind between episodes and the action space with them.
- `costs` maps `maker`, `taker`, `slippage` and `impact` to `(low, high)` ranges, a fee as a rate, slippage in basis points and impact as the coefficient of the venue's impact model, and each environment draws its value of each uniformly once, from the stream named `costs` of the seed, keeping it for its life: environment `i` takes the `i`-th four draws, one for each cost in that order whether it is named or not, so its costs are the same whatever `num_envs` is. A cost not named is the venue's, and a fee drawn is every instrument's in place of its own.
- Each environment `i` draws its episodes from the stream named `episodes i` of the seed, each episode its start and then its slots, so its `k`-th episode is the same whatever `num_envs` is; a seed left out is drawn once and recorded, so the episodes come again from it. `reset(seed=)` starts the stream of each environment it resets again from that seed, and leaves the costs as they were drawn. Under `intrabar="random"` each environment decides the bars' hidden events from a stream of its own, as a batch's do ([What a bar cannot show](#what-a-bar-cannot-show)).

### The observation

The observation is float32, the market block and then the account block, laid out flat by default or as a dict of the two with `layout="dict"`.
- **Market**, `(W, N, F)` over the window's bars, its slots and their features: a shown instrument's features on the window's bars it printed, and 0.0 on those it did not. With the candles as features, a bar's open, high, low and close are each over the instrument's last close, and its volume is over the mean volume of the window's bars it printed, 0.0 where that mean is zero.
- **Account**, for each slot in turn its weight, its notional at its last close over equity as [Target](#target) reckons it; its unrealized PnL over equity; its last close over its average entry, less one, 0.0 when flat; and 1.0 where it is shown, else 0.0. Then the account's log return over the step as the reward reads it, 0.0 at an episode's first observation; its leverage, gross notional over equity; its margin ratio, the maintenance its perpetuals keep over the margin wallet's equity, 0.0 where nothing is kept and 1.0 where something is and that equity is at or below zero; and its cash weight, one less the sum of the weights.
- With `actions="orders"` each slot adds the number of its orders resting and the nearest one's distance from its last close in ticks, 0.0 where none rests.
- An instrument not shown has every market cell 0.0 and its action ignored; its weight, PnL and distance still read its position at its last close. A value over equity is 0.0 where equity is at or below zero, so no cell is ever NaN or infinite.
- With `normalise=True` each market value of feature `g` is written as `(x - m_g) / d_g`, where `m_g` and `d_g` are the mean and the deviation of every value of feature `g` the training span's observations hold: at each bar at which an instrument is shown, each bar of its window it printed, and for the candles' volume only a window whose mean volume is above zero. A feature whose values lie within a billionth of their size of one another, `max - min <= 10^-9 * max(|min|, |max|)`, never varies, and takes its smallest value as `m_g` and `d_g = 1`; one that holds no value takes `m_g = 0` and `d_g = 1`, as does a deviation whose squares underflow to zero; and a cell the observation leaves at 0.0 stays 0.0.

### Actions

The decision at a step is a [Target](#target) of each shown slot's weight, with the environment's `tolerance`, the slots not shown going unnamed; or, for `"orders"`, an order for each.
- `"box"`, the default: `Box(-1, 1, (N,))`, a spot slot's lower bound 0. The actions are clipped to the space and each weight is `max_leverage * a / max(1, sum |a|)`, the sum over the shown slots, so the gross weight never exceeds `max_leverage`.
- `"simplex"`: `Box(0, 1, (N + 1,))`, the last a cash slot, each weight its action over the sum of the shown slots' and the cash slot's, all cash where they sum to zero.
- `"discrete"`: each slot short, flat or long at `max_leverage / N`, a spot slot flat or long, as a `Discrete` for one slot and a `MultiDiscrete` for several.
- `"orders"`: `Box` of `(N, 2)`, a slot's size in `[-1, 1]` and its offset in `[0, 1]`. The size is a limit order's notional, `|size| * max_leverage * equity` at the last close, a buy where it is positive and a sell where negative, and the offset puts its price `round(offset * max_offset)` ticks from the last close on the passive side, a half rounded to the even number of ticks. Each step's order for a slot replaces the one the last step left resting, a size of zero places none, as does a buy whose price would not be above zero, and every slot needs a tick.
- `decode(actions, state)` returns the `(B, N)` weights itself, once a step for the whole batch, over the same state a reward reads; a NaN weight leaves its slot unnamed.

### Reward and termination

- The floor is 1% of the opening balance, both wallets together. An episode terminates on a bar where the account is liquidated or its equity falls to the floor, whatever the reward, so the equity before every rewarded step is above the floor.
- `"log"`, the default: `log(max(E_t, floor) / E_(t-1))`, a liquidation a large finite penalty.
- `"sharpe"`: the differential Sharpe of those log returns `R_t`, at the rate `eta = 2 / (W + 1)` of an average over the window, from moments `A` and `B` that start an episode at zero: `D_t = (B * dA - A * dB / 2) / (B - A^2)^1.5` with `dA = R_t - A` and `dB = R_t^2 - B` read before the moments move, 0.0 where `B - A^2` is not above zero; then `A += eta * dA` and `B += eta * dB`. A window of one bar, at a rate of 1, keeps the last return alone and so no deviation, and is refused.
- `reward(state)` returns the `(B,)` rewards itself, once a step for the whole batch. Its `state` holds `(B,)` arrays of `equity`, `previous_equity`, `log_return`, the log return above; `drawdown`, one less equity over the episode's highest close of equity, the opening balance included; `leverage`; `turnover`, the bar's traded notional over the previous equity; `fees` and `funding`, the bar's over the previous equity, funding positive where paid; and `liquidated`; and `(B, N)` arrays of each slot's `weights` and `positions`.

### Autoreset

- `"next_step"`, the default: the step after an episode ends resets that environment, ignoring its action, and returns its first observation with a reward of 0.0 and neither flag set.
- `"same_step"`: the step that ends an episode resets it and returns the next episode's first observation, the last one and its info under `final_obs` and `final_info`.
- `"disabled"`: an episode that ends is left as it ended, each later step ignoring its action and returning its last observation and flags with a reward of 0.0, until `reset(options={"reset_mask": mask})` resets the environments `mask` names.
- Every step's info holds each environment's `episode`, its number in its stream, its `step` and its `start`, the Data's row it opened on, and with `sample` its `instruments`, so any episode can be found again and run as a backtest from its start.

### What an environment says

`env.describe()` lists the assumptions its runs make as [Assumptions](#assumptions) lists a backtest's at the venue, a cost drawn per environment shown with its range, and `env.provenance` records the Data's provenance, the venue, every setting, the seed and the streams each environment draws from.

**Worked example.** With one instrument, the candles as features and a window of 64, the observation holds `64 * 1 * 5` market values and `4 * 1 + 4` account values, 328.

**Worked example.** A bar opening at 100 with a high of 110, a low of 95 and a close of 105, the instrument's last close, and a volume of 300 where the window's printed bars average 200, reads `(0.952381, 1.047619, 0.904762, 1.0, 1.5)`.

**Worked example.** At `max_leverage=2.0` a single slot's action of 0.3 is a weight of 0.6, and two slots' actions of 0.8 and -0.6, summing to 1.4 in absolute value, are weights of `2 * 0.8 / 1.4 = 1.142857` and `-0.857143`.

**Worked example.** With a balance of 10,000 the floor is 100, so a takeover that leaves no equity after a step from 2,000 is rewarded `log(100 / 2,000) = -3.00` and ends the episode; a sale on the book that leaves 1,500 is rewarded `log(1,500 / 2,000) = -0.29` and ends it too.

**Worked example.** At `max_leverage=2.0`, an equity of 10,000 and a last close of 50,000 with a tick of 0.1, an order action of 0.25 with an offset of 0.5 and `max_offset=20` is a buy limit of `0.25 * 2 * 10,000 / 50,000 = 0.1` BTC at `50,000 - 10 * 0.1 = 49,999.0`.

**Worked example.** With `normalise=True`, a feature whose values in the training span's observations average 0.5 with a deviation of 2.0 is written as `(3.0 - 0.5) / 2.0 = 1.25` where it is 3.0.

### The agent spec

`env.agent_spec` is a JSON document, schema 1, of what a backtest needs to show an agent what it saw in training and to read its outputs as they were read ([ADR 0039](Decisions.md)).
- The inputs: `window`; `instruments`, the Data's names, with `markets`, each one's market as the venue gave it, `slots` and `sample`; `features`, `{"builtin": "candles"}`, `{"array": F}` for an array, or for a function `{"custom": name, "width": F, "source_hash": s, "reference": r}`; `fill`; `normalisation`, null or `{"mean": [m_g], "deviation": [d_g]}`; and `sb3`, null, or the statistics SB3's `VecNormalize` kept of the observation, its `mean` and `var`, by key for a dict, with its `clip` and `epsilon`, which `probatum.sb3.SpecCallback` copies into the spec on the model's action space after each rollout, having taken the spec of the environment it trains on as the run of training started; they are the model's own, and an environment's spec holds none.
- The observation: its `size`, its `layout`, the market block's `shape` as `[W, N, F]`, and its `account` values by name, for each slot `weight`, `upnl`, `entry_distance` and `shown`, with `resting` and `nearest` for orders, then `log_return`, `leverage`, `margin_ratio` and `cash_weight`.
- The outputs: `actions`, the space with its bounds, `max_leverage` where it scales the weights, and `max_offset` for orders; `decode`, `{"builtin": kind}` or a custom decode's name, source hash and reference; `reward`, `{"builtin": "log", "floor": 0.01}`, the Sharpe's with its `eta`, or a custom reward's name and source hash; and `tolerance`.
- Where it trained: `venue`, the `name` of the preset the venue was made from, null for a venue built by hand, the day that preset's tables were `taken`, and every setting `changed` from where it started, as the venue's repr shows them and each float JSON cannot hold written as `{"float": "inf"}`, so the venue is that preset's variant, or a venue built by hand, with those settings; `training`, the Data's `fingerprint`, its `interval`, its number of `bars` and its first bar's time `start` and its last's `end`; and the `version` of probatum.
- A source hash is blake2b's 8 bytes over the function's source text, written as `"b2:"` and 16 hex characters, and null where the source cannot be read. A reference is the same over the shape of the function's output on its probe, each dimension a little-endian 64-bit integer, and the output in order, each value rounded to a little-endian float32, every NaN one NaN and every zero +0.0, so a function passed again is refused where its probe output differs once rounded. A function that raises on its probe, or gives no `(512, N, F)` array there, leaves its reference null, with a warning, and a function passed again is held to nothing but its width. Passed again, a function is held by its kind too: the candles take none, an array takes an array of its width, a function a function, a builtin decode none and a custom one a function.
- A feature function's probe is 512 bars of each training instrument, under its name and on its market, with no contract, quote or base, on the training interval from 2024-01-01 00:00 UTC; a function that gives another width there than on the training bars leaves its reference null. Instrument `i` draws from `x_(n+1) = (6364136223846793005 * x_n + 1442695040888963407) mod 2^64` from `x_0 = i + 1`, each draw `u = floor(x / 2^11) / 2^53`, four a bar, `r`, `h`, `l` and `v` in turn. Its first bar opens at 100 and each later one at the last close; the close is the open times `1 + 0.02 * (r - 0.5)`, the high `max(open, close) * (1 + 0.005 * h)`, the low `min(open, close) * (1 - 0.005 * l)` and the volume `1,000 * (1 + v)`. Plain arithmetic alone builds them, so they are the same on every machine.
- A decode's probe is 8 rows of actions over a fresh flat account, every equity 10,000 and every other value of the state zero: each value of each row in turn, the space's components in order, draws from the same generator from `x_0 = 2^32`, a box component `low + (high - low) * u` and a simplex one `u`, as float64, and a discrete one `floor(u * n)` of its `n` choices, as int64. A decode's reference is taken the first time the spec is read, so a decode that keeps state of its own meets its probe then, and a decode passed again meets it once more as it is held to the spec.

**Worked example.** Instrument 0's first probe bar opens at 100 and draws `r = 0.423209`, so it closes at `100 * (1 + 0.02 * (0.423209 - 0.5)) = 99.846418`, with a high of 100.254704, a low of 99.522737 and a volume of 1,382.863391.

### An agent in a backtest

`backtest(agent, data, venue, record="full", risk_free=0.0, spec=None, features=None, decode=None, instruments=None)` runs a trained agent as [An array](#an-array) and [A strategy](#a-strategy) run a policy ([ADR 0039](Decisions.md)).
- The agent is an SB3 model or policy, whose action is its `predict(observation, state, episode_start, deterministic=True)`, a recurrent policy's state carried from one close to the next; a torch module, whose action is its output for the observation as a tensor on the module's device, worked in eval mode with no gradient, its mode restored after; or any other function of the observation. Where a module or a function returns a tuple, its first element is the action. `spec` is the agent spec, a path to its JSON or to a directory holding `spec.json`, or the environment the agent trained in, and an SB3 model's is the one on its action space where `spec` is left out; an SB3 model or policy that carries a spec reads its `sb3` statistics from it whatever `spec` is given.
- The slots are the Data's instruments named by `instruments`, or every one of its instruments where it is left out, and must be the spec's `slots` in number; an unsampled spec's slots must be its `instruments`, in order, each the same symbol, whether or not either Data names it with its market or venue before it, and every slot must be on the market the agent trained on, as the venue gives it. `features` and `decode` are held to the spec as [The agent spec](#the-agent-spec) says, an array of features is over the Data given, the slots' taken from it, as are a sampled agent's features from a function, which training worked over the universe it sampled from, and a feature function that gives another number of features a bar than the spec's `shape` is refused.
- The run is one environment of the spec's settings on the slots' bars at `venue`, recording at `record`, from the Data's first bar to its last: the window reads the bars before a slice, an instrument's features starting after the last bar it printed with one that is not finite in float32, so an instrument shown at the slice's first bar is decided on at its close. Each close's observation is written as in training, normalised by the spec's `normalisation` and, where the spec's `sb3` holds the statistics SB3's `VecNormalize` kept, `clip((x - mean) / sqrt(var + epsilon), -clip, clip)` as float32 over the observation, or over each key of a dict it normalised; the agent's action is decoded as the spec says and placed as a step places it. Once the account's equity is at or below zero the agent is no longer asked, and each slot is left unnamed, or with orders given no order.
- The Result is the run's, `result.policy` the decisions, each close's weights or, for orders, each slot's size and offset as placed, clipped to their bounds, NaN where nothing was decided or a slot was not shown. Its frictionless baseline is the agent run again without the costs of trading, at once at the full level rather than when the baselines are first read, as an agent cannot be copied as it stands to be played later. A run on bars that overlap the training span, at a venue the spec does not name, or on another interval warns, and the Result's provenance records the spec, the overlap as the first and the last time both hold, and whether the venue is the one the agent trained at. [Capacity](#capacity) and the [cost sweeps](#cost-sweeps) run an agent with these settings at each level, with no baseline.

**Worked example.** With SB3's statistics of a mean of 1.0, a variance of 0.04, an epsilon of 1e-8 and a clip of 10, an observation value of 9.0 is given to the agent as `min((9.0 - 1.0) / sqrt(0.04 + 1e-8), 10) = 10.0`, and one of 1.5 as `0.5 / sqrt(0.04000001) = 2.4999998` once rounded to float32.

### Tensors on a device

`probatum.torch.TensorEnv(env, device="cpu", resident=False)` wraps the environment `probatum.env` returns and hands out what each `reset` and `step` returns as torch tensors on `device`: the observation as float32, flat or as a dict of its two blocks, the rewards as float64 and the terminated and truncated flags as booleans, each the numpy path's to the bit; the info stays as the environment gives it ([ADR 0040](Decisions.md)).
- Actions, and a reset's `reset_mask`, are a tensor on any device or anything numpy reads, and reach the environment on the host, one synchronisation a step.
- On a CUDA device the observation is written into the first or the second of two pinned host buffers in turn, and the rewards and flags are copied into pinned buffers of their own; each is copied to the device without blocking the host, and an event recorded after a turn's copies marks when its buffers are free, which the step after next waits for before writing them again. Every tensor handed out is a fresh device tensor, so no later step changes one. On any other device the copies are synchronous.
- With `resident=True` the market block is worked on the device. At the wrap the market's cells are copied there once: with features, each bar's cells as the observation writes them, over the scale and rounded to float32, and 0.0 where its instrument did not print; with the candles, the bars in float64 and each bar's mean volume over the printed bars of the window ending at it, summed in order from the window's first bar, as the engine sums it. At each step the host gives each environment's row `t`, its slots and its account block, and the device gathers rows `t - W + 1` to `t` of each slot: with the candles, each bar's open, high, low and close over the close at `t` and its volume over the mean at `t`, 0.0 where that mean is not above zero, each over the scale in float64 and then rounded to float32. Every cell of a bar its instrument did not print, and of a slot the account block says is not shown, is 0.0.
- The resident mode is refused under `same_step` autoreset, whose final observations are written on the host.
- A device given without an index, such as `"cuda"`, is the one current at the wrap, which every step's tensors are handed out on.
- The wrap takes the environment over: from then on it writes into TensorEnv's buffers whatever its `copy`, which its provenance keeps as given. A copy of a TensorEnv wraps a clone of its environment as one of its own.

**Worked example.** One step of 4,096 environments of ten instruments, eight features and a 64-bar window copies `4,096 x (64 x 10 x 8 + 4 x 10 + 4) x 4 = 84,606,976` bytes of observation to the device on the host path, and `4,096 x (8 + 10 x 8 + 44 x 4) = 1,081,344` bytes in the resident mode: each environment's row and its ten slots as 64-bit integers, and its 44 account values.

### Clones and pickles

`engine.clone()`, `batch.clone()` and `env.clone()` copy a `probatum.Engine`, a `probatum.rl.Batch` or an environment as it stands, sharing its Data's bars and its features and copying everything else: each account, its resting orders and its cursor, what it has recorded, every random stream it draws from, an environment's spaces with their generators, and each episode's counters, slots, start and the reward's running values, so the copy and the original, given the same actions, return the same steps to the bit ([ADR 0040](Decisions.md)).
- `batch.clone(env=i, n=k)` and `env.clone(env=i, n=k)` fork environment `i` into `k` environments, each a copy of it, `n` one where it is left out; each draws its next episodes, and under `intrabar="random"` each bar's hidden events, from copies of `i`'s streams, so the fork's environments repeat one another for the same actions; `reset(seed=)` starts each one's episodes again, as it does any environment's, and leaves its hidden events drawing on from its copy.
- A copy of an environment keeps the transitions it recorded, and a fork starts with none.
- A copy of `probatum.Engine` hands out handles of its own for the orders placed so far, and refuses the original's.
- `pickle` moves a `probatum.Engine` or a `probatum.rl.Batch` between processes of one version, writing its whole state and the bars with it; a process of another version refuses it, naming both. It is not a way to keep a run, which a Result's files are.
- An environment does not pickle, since it holds the functions it was given; it is built in the process that steps it.

### Transitions

`record="transitions"` keeps one row for each step an environment takes in an episode ([ADR 0040](Decisions.md)):
- `env`, the environment's index; `episode`, its episodes counted from 0 in the order it began them, which `reset(seed=)` does not start again as it does the info's, so `env` and `episode` name one episode; `step`, the steps its episode had taken before this one; `observation`, the observation the environment wrote at the close the action was decided at, flat, the market block and then the account block whatever the layout, and without the statistics of SB3's `VecNormalize` that an agent's backtest applies; `action`, the action as it was given, before it is decoded, flat, as float32, or as int64 for a discrete action; `reward`; `terminated` and `truncated`; and `next_observation`, the observation the step returned, or under `same_step` the last of an episode it ended.
- A step that resets an environment under `next_step` and ignores its action takes no row, and nor does a step that leaves an ended episode as it was under `disabled`. A step whose `reward` function raises takes none either: each environment it left in an episode opens its next row on the bar the step reached, and under `same_step` one whose episode it ended starts the next at the step after, which takes no row, as under `next_step`. A row's flags are the environment's own: an episode a wrapper such as Gymnasium's `TimeLimit` cuts, or a `reset` ends early, closes on a row with neither flag set, and its `episode` tells it from the next.
- `env(..., record="transitions")` records every step its environments take, and `env.transitions` hands them out. `TensorEnv`'s resident mode is refused on such an environment, whose observations are kept whole on the host.
- `backtest(agent, ..., record="transitions")` keeps the account level's records and, as `result.transitions`, a row for each close the agent is asked at until its episode ends, environment 0 and episode 0, rewarded with the spec's `log` or `sharpe` reward. An agent whose spec records a reward of its own is refused, as a backtest has no function to work it, and its transitions are recorded in its environment; a policy that is not an agent takes no steps, and is refused at this level.
- The transitions are a table of those columns, each a numpy array, `observation` and `next_observation` `(R, D)`, `action` `(R, A)` and the rest `(R,)`, with their `metadata`: the `spaces`, the spec's `observation` and `actions`; the `reward` as the spec records it; the `spec`, an environment's as its `agent_spec` gives it when they are read, or the one an agent's backtest ran under; and probatum's `version`.
- `export("arrow", path)`, on a Result or an environment, writes them as one Arrow IPC file at `path`: `env`, `episode` and `step` as int64, `observation`, `action` and `next_observation` as fixed-size lists of float32, or of int64 for a discrete action, `reward` as float64 and the flags as booleans, with the metadata on its schema as JSON text under `probatum.spaces`, `probatum.reward`, `probatum.spec` and `probatum.version`.

**Worked example.** Under `next_step` an episode that ends at its third step gives rows of steps 0, 1 and 2, the last terminated or truncated, and the step after it, which resets the environment, gives none; the new episode's first row is step 0 of the next episode.

**Worked example.** With one instrument's candles over a 64-bar window an observation is 328 floats, so a row holds `2 x 328 x 4 = 2,624` bytes of observations beside a 4-byte action, three 8-byte integers, an 8-byte reward and two flags.

<br>
<br>


## Numbers that are the same everywhere

- Basic arithmetic and `sqrt` are correctly rounded everywhere, but `ln`, `exp` and `powf` may differ by platform, so integer powers are written as products and every other transcendental function goes through the `libm` crate.
- Random draws come from named streams, PCG64 in Python and ChaCha in Rust, each derived from the run's seed through a NumPy `SeedSequence` and recorded with it; a search's random and grid samplers draw from them, a log scale's `ln` and `exp` taken through `libm`. A statistic's resampling, and the clustering of a search's trials, draw from a named stream seeded by a digest of the values they read instead, so they repeat with no seed ([Inference](#inference), [Across a search](#across-a-search)).
- Sums run in a fixed instrument order, the batch has no cross-environment reductions, and the statistics a search ranks on are summed in a fixed order.
- Compensated sums are used only for totals reported over many bars, such as fees, funding and turnover; cash and positions stay plain arithmetic in event order.
- A perpetual fill that leaves no perpetual held, with the margin wallet's size within `1e-12` times the money at stake, leaves the wallet at zero, as a takeover does, the remainder booked as the liquidation charge. The money at stake is what the bar's perpetual fills have put there so far, for each the margin wallet before it, what the margin bound paid in and its notional. Exact arithmetic leaves zero where the margin ran out, and floating point a hair either side of it, which above zero is equity the cap would lever.

<br>
<br>


## The low-level binding

Below the Python surface, one binding drives the engine directly: it takes the candles, a venue record and a per-bar action script, and returns the ledger, the fill log, the order log and the round trips. A cancel or a replacement names its order by a reference resolved at run time, an earlier placement in the script or the oldest order still live, so a refused placement leaves the script meaningful. The reference simulator and the tests drive the engine through it. An array run goes through the engine in one call too, which takes its rows and its barriers and decides each close itself.
