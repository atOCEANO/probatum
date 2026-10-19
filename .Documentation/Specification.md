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

This page is every rule the engine applies to money: when an order fills and at what price, what each fill costs, when funding is charged, how margin is measured, where an account is liquidated, what a run records, how a search scores and checks its trials, and how a validation divides the bars and runs its paths. Each rule is stated as an equation where it has one and followed by a worked example, and every worked example is a test, so a number on this page is a number the engine produces. Why each rule is the way it is lives in [Decisions](Decisions.md).

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

`"random"` draws one of the two for each event from the stream named `intrabar` in the run's seed, which in a batch is named with the environment's index as well. Every bar draws in one order whether or not each event arises: the instruments together, the exits held back, each instrument's exits, each instrument's fill inside the bar, each funding event the row owns, then the backstop. What a bar draws therefore never depends on what the account holds. The stream runs on across a reset, so each episode draws afresh. `backtest`, `search` and `validate` refuse it.

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

`[a]` is 1 when `a` holds and 0 otherwise. The estimate is undefined with fewer than three bars, with fewer than two pairs where `tau` is 1, or with `po` or `pc` at zero, and an undefined estimate is a spread of zero. So is a negative `s2`, where the authors' published code returns the root of its magnitude.

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
| delisting | "no delisting time; ETHUSDT's bars end 2025-03-01 12:00", where an instrument's bars end before the last row | simplified | either |
| universe | "the user's list of 3 instruments" | simplified | optimistic |
| prices | "as the venue printed them", the router's bars | venue | |
| prices | "as given; splits, dividends and borrow costs only as the user adjusted them" | simplified | either |

A Data's bars are the router's where every instrument names its venue, and they read as another venue's where the run's venue is a preset of a different exchange.

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
| `"transitions"` | what offline reinforcement learning needs; not built yet |
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

<br>
<br>


## Validation

`validate` measures how a search over a Strategy's parameters holds up across folds. It runs a search over the bars before the holdout, whose winner the holdout runs, then the same search on each split's training bars, and trades each split's winner on its test span; the test spans join into out-of-sample paths, each one run of the engine.

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

- A split's spans are its runs of adjacent test groups, one span for WalkForward and Holdout. A span from bar `a` to bar `b` reaches `L`, the larger of `b` and the latest end among its rows that have a label.
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

### Paths and seams

- A path's windows are its runs of groups, or its WalkForward or Holdout spans, that one split's winner trades; a group with no row a run decides at, a printed row before the last, makes no window. A window decides at the closes of its printed rows `f` to `l` and owns what those decisions govern: the returns, fills, closed round trips, bars in a position and costs of the account's rows from the one after `f` to the one after `l`, the last window's to the end.
- A path is one run, over the in-sample bars, of a composite Strategy whose account opens at their first printed row. Its `init` builds each window's Strategy with its fit's parameters, initialises each on the in-sample bars and reads each one's warm-up; each decides only at its own window's closes and from its own warm-up. A path's `warmup` is its first window's first row, so nothing trades before it.
- At the close of a window's first decision row, that bar having filled what the window before it left: every order not yet filled is cancelled, the waiting market and value orders in the order they were placed and then the resting ones by slot, each with the reason `seam`; every target is forgotten, so an instrument records no drift until a target names it again; the decision's events start empty, and the account's round trips and equity curve start from that close; `on_inherit(bar, account)` is called; and then `next`. A path's first window has no seam.
- With `seam="flatten"` the last decision row of each window but a path's last closes every position instead of calling `next`: each order not yet filled is cancelled as above, and a reduce-only `ioc` market order closes each instrument's position, in the instruments' order, placed and filled with the reason `seam`. What it leaves open, for want of liquidity, a size limit or a bar, is the next window's. Under `flatten` a window with another after it holds at least two printed rows.
- A window's statistics are the engine's over its rows, opening at the equity of the close before them, and a path's over the rows from its first window's first decision on. A path whose run raises, or that a failed fit trades, fails, keeping its error, and `validate` raises where no path completes.

**Worked example.** Window 1 decides at the closes of rows 0 to 9 and window 2 from row 10. At row 9's close window 1 holds a long of 1 BTC with a sell stop resting at 95, and buys ETH, which prints no bar at row 10. Under `cancel`, at row 10's close the stop and the ETH buy still waiting are cancelled with the reason `seam`, window 2's `on_inherit` finds the long of 1 and nothing resting, and its `next` runs; window 1 owns rows 1 to 10 and window 2 rows 11 on. Under `flatten`, window 1 does not decide at row 9: its stop is cancelled there, and a sell of 1 BTC placed there fills at row 10's open with the reason `seam`, so window 2 inherits no position.

### The Study of a validation

- `paths`, a row a path: its number, state, score, the 15 statistics and the seven costs of the rows it owns, and its error. `fits`, a row a split: its number, what it tests, its training, purged and embargoed bars, its pieces, its winner's parameters and fitted score, its trials run and complete, its seed, state and error. `windows`, a row a path's window: its path and fit, the first and last account rows it owns and their times, the rows it decided on after its warm-up, its score, statistics and costs.
- `returns` holds each path's returns on the in-sample account rows, `(paths, rows)`, NaN where the path owns no row and on a failed path, the columns a search's on the same bars; `keep="none"` keeps none.
- `best`, `best_params`, `best_score`, `evaluate_holdout`, `holdout` and `looks` are the first search's ([The holdout](#the-holdout)).

<br>
<br>


## Numbers that are the same everywhere

- Basic arithmetic and `sqrt` are correctly rounded everywhere, but `ln`, `exp` and `powf` may differ by platform, so integer powers are written as products and every other transcendental function goes through the `libm` crate.
- Random draws come from named streams, PCG64 in Python and ChaCha in Rust, each derived from the run's seed through a NumPy `SeedSequence` and recorded with it; a search's random and grid samplers draw from them, a log scale's `ln` and `exp` taken through `libm`.
- Sums run in a fixed instrument order, the batch has no cross-environment reductions, and the statistics a search ranks on are summed in a fixed order.
- Compensated sums are used only for totals reported over many bars, such as fees, funding and turnover; cash and positions stay plain arithmetic in event order.
- A perpetual fill that leaves no perpetual held, with the margin wallet's size within `1e-12` times the money at stake, leaves the wallet at zero, as a takeover does, the remainder booked as the liquidation charge. The money at stake is what the bar's perpetual fills have put there so far, for each the margin wallet before it, what the margin bound paid in and its notional. Exact arithmetic leaves zero where the margin ran out, and floating point a hair either side of it, which above zero is equity the cap would lever.

<br>
<br>


## The low-level binding

Below the Python surface, one binding drives the engine directly: it takes the candles, a venue record and a per-bar action script, and returns the ledger, the fill log, the order log and the round trips. A cancel or a replacement names its order by a reference resolved at run time, an earlier placement in the script or the oldest order still live, so a refused placement leaves the script meaningful. The reference simulator and the tests drive the engine through it. An array run goes through the engine in one call too, which takes its rows and its barriers and decides each close itself.
