"""The environment as the Specification states it, worked in plain Python: an episode
of one environment over a universe, run through a reference simulator opened at its
start, its observation, its decoding, its reward and its flags read from the account
that simulator keeps (Specification, The environment). It knows nothing of the batch,
which test_the_environment_agrees_with_the_reference.py holds to it. Beside it are the
agent spec's statistics, probes and hashes, worked as plain loops (Specification, The
agent spec).
"""

import hashlib
import math
import types

import numpy as np

# the agent spec's generator, x_(n+1) = (A * x_n + C) mod 2^64
A, C = 6364136223846793005, 1442695040888963407


def logged(equity, before, floor):
    """Reward and termination: the step's log return, its equity floored."""
    return math.log(max(equity, floor) / before)


def limit(size, offset, equity, mark, inst, max_leverage, max_offset):
    """Actions, orders: the side, amount and price of the limit a slot's action asks
    of instrument `inst` marked at `mark`, None for a size of zero or a buy whose price
    would not be above zero."""
    size   = min(max(float(size), -1.0), 1.0)
    offset = min(max(float(offset), 0.0), 1.0)
    if size == 0.0:
        return None

    notional = abs(size) * max_leverage * equity
    amount   = (notional * mark / inst.contract if inst.kind == "inverse"
                else notional / mark)

    ticks = round(offset * max_offset)
    if size > 0.0:
        price = mark - ticks * inst.tick
        return ("buy", amount, price) if price > 0.0 else None

    return "sell", amount, mark + ticks * inst.tick


def onsets(columns, features):
    """Each instrument's `e`: the first bar it printed on with every feature finite,
    its first printed bar where `features` is None and the features are its candles;
    None where it has none."""
    found = []
    for i, column in enumerate(columns):
        e = None
        for t, bar in enumerate(column):
            if bar is not None and (features is None
                                    or all(math.isfinite(x) for x in features[t][i])):
                e = t
                break
        found.append(e)

    return found


def moments(columns, features, window, shown):
    """The Observation, normalise=True: each feature's mean and deviation over every
    value the observations hold, at each bar `t` instrument `i` is shown at, by
    `shown(i, t)`, each bar of its window it printed, the candles' volume only where
    the window's mean volume is above zero; a feature whose values lie within a
    billionth of their size of one another never varies, and takes its smallest value
    and a deviation of 1, and one of no value a mean of 0 and a deviation of 1."""
    held  = {}
    width = 5 if features is None else len(features[0][0])

    for i, column in enumerate(columns):
        for t, bar in enumerate(column):
            if not shown(i, t):
                continue
            bars = [r for r in range(t - window + 1, t + 1) if column[r] is not None]
            if features is not None:
                for r in bars:
                    for g, x in enumerate(features[r][i]):
                        held.setdefault(g, []).append(float(x))
                continue

            mean = sum(column[r].volume for r in bars) / len(bars)
            for r in bars:
                one = column[r]
                for g, x in enumerate((one.open, one.high, one.low, one.close)):
                    held.setdefault(g, []).append(x / bar.close)
                if mean > 0.0:
                    held.setdefault(4, []).append(one.volume / mean)

    out = []
    for g in range(width):
        values = held.get(g, [])
        if not values:
            out.append((0.0, 1.0))
            continue

        low, high = min(values), max(values)
        if high - low <= 1e-9 * max(abs(low), abs(high)):
            out.append((low, 1.0))
            continue

        m = sum(values) / len(values)
        d = math.sqrt(sum((x - m) ** 2 for x in values) / len(values))
        # a spread whose squares underflow to zero is no spread to divide by
        out.append((m, d if d > 0.0 else 1.0))

    return out


def draws(x, count):
    """The agent spec: `count` draws of the probe generator after state `x`, and the
    state it leaves."""
    out = []
    for _ in range(count):
        x = (A * x + C) % 2**64
        out.append((x >> 11) / 2**53)

    return out, x


def probe(width, rows=512):
    """The agent spec: a feature function's probe, each of `width` instruments' bars as
    (open, high, low, close, volume)."""
    columns = []
    for i in range(width):
        x, bars, opened = i + 1, [], 100.0
        for _ in range(rows):
            (r, h, l, v), x = draws(x, 4)
            close = opened * (1.0 + 0.02 * (r - 0.5))
            bars.append((opened, max(opened, close) * (1.0 + 0.005 * h),
                         min(opened, close) * (1.0 - 0.005 * l), close,
                         1000.0 * (1.0 + v)))
            opened = close
        columns.append(bars)

    return columns


def probe_actions(space):
    """The agent spec: a decode's probe, 8 rows of actions, each over `space`, its
    components in order as ("box", low, high), ("simplex",) or ("discrete", n)."""
    rows, x = [], 2**32
    for _ in range(8):
        row = []
        for component in space:
            (u,), x = draws(x, 1)
            if component[0] == "box":
                row.append(component[1] + (component[2] - component[1]) * u)
            elif component[0] == "simplex":
                row.append(u)
            else:
                row.append(float(math.floor(u * component[1])))
        rows.append(row)

    return rows


def digest(shape, values):
    """The agent spec: a reference, blake2b's 8 bytes over `shape`, each a
    little-endian 64-bit integer, and `values` in order, each rounded to a
    little-endian float32, every NaN one NaN and every zero +0.0."""
    out = hashlib.blake2b(digest_size=8)
    for n in shape:
        out.update(int(n).to_bytes(8, "little", signed=True))

    for x in values:
        # a value past float32's range rounds to an infinity, as numpy's cast does
        with np.errstate(over="ignore"):
            y = np.float32(x)
        y = np.float32(np.nan) if np.isnan(y) else (np.float32(0.0) if y == 0.0 else y)
        out.update(np.array([y], dtype="<f4").tobytes())

    return "b2:" + out.hexdigest()


class Episode:
    """One episode of one environment: `make(start)`, the simulator opened at `start`,
    its `slots`, the instruments it trades, and what it keeps between steps.
    `settings` holds `columns`, each instrument's bars, None where it printed none;
    `features`, rows of each instrument's features, or None for the candles; `window`,
    `actions`, `max_leverage`, `max_offset`, `reward`, `tolerance`, `episode_length`
    and `balance`, the opening balance of both wallets."""


    def __init__(self, make, start, slots, settings):
        self.sim      = make(start)
        self.slots    = list(slots)
        self.s        = types.SimpleNamespace(**settings)
        self.e        = onsets(self.s.columns, self.s.features)
        self.floor    = 0.01 * self.s.balance
        self.steps    = 0
        self.previous = self.sim.equity()
        self.peak     = max(self.s.balance, self.previous)
        self.moments  = [0.0, 0.0]

        # the last step's log return, 0.0 at the first observation
        self.returned     = 0.0
        self.ordered      = {}   # each slot's order the last step left, by its id
        self.ended        = False
        self.liquidations = self.sim.liquidations
        # what the last step's reward read, which the next decision's decode reads
        self.last = self.state(self.previous, [], 0.0)


    def printed(self, i, t):
        return self.s.columns[i][t] is not None


    def shown(self, i):
        """Episodes: entered, printed at the current bar and not delisted."""
        t, e = self.sim.t, self.e[i]
        return (e is not None and t - self.s.window + 1 >= e and self.printed(i, t)
                and not self.sim.delisted[i])


    def market(self, i):
        """The Observation's market block of slot instrument `i`: its window's rows of
        features, 0.0 on a bar it did not print and everywhere when it is not shown."""
        w, t = self.s.window, self.sim.t
        width = 5 if self.s.features is None else len(self.s.features[t][i])
        rows  = [[0.0] * width for _ in range(w)]
        if not self.shown(i):
            return rows

        bars = range(t - w + 1, t + 1)
        if self.s.features is not None:
            for k, r in enumerate(bars):
                if self.printed(i, r):
                    rows[k] = [self.scaled(g, float(x))
                               for g, x in enumerate(self.s.features[r][i])]
            return rows

        last    = self.s.columns[i][t].close
        printed = [self.s.columns[i][r] for r in bars if self.printed(i, r)]
        mean    = sum(bar.volume for bar in printed) / len(printed)

        for k, r in enumerate(bars):
            bar = self.s.columns[i][r]
            if bar is not None:
                rows[k] = [self.scaled(g, x / last) for g, x in
                           enumerate((bar.open, bar.high, bar.low, bar.close))]
                rows[k].append(self.scaled(4, bar.volume / mean) if mean > 0.0 else 0.0)

        return rows


    def scaled(self, g, x):
        """The Observation: feature `g`'s value `x` as it is written, over the mean
        and the deviation `scale` gives it with normalise=True."""
        scale = getattr(self.s, "scale", None)
        return x if scale is None else (x - scale[g][0]) / scale[g][1]


    def weight(self, i, equity):
        q, mark = self.sim.position[i], self.sim.mark[i]
        if q == 0.0 or equity <= 0.0:
            return 0.0
        return self.sim.instruments[i].notional(q, mark) / equity


    def account(self):
        """The Observation's account block: each slot's four, or six with orders, then
        the account's four."""
        sim    = self.sim
        equity = sim.equity()
        row    = sim.ledger[-1]
        out, weights = [], []

        for i in self.slots:
            inst, q, mark = sim.instruments[i], sim.position[i], sim.mark[i]
            weight = self.weight(i, equity)
            weights.append(weight)
            held = q != 0.0
            upnl = (inst.profit(q, sim.entry[i], mark) / equity
                    if held and equity > 0.0 else 0.0)
            out += [weight, upnl, mark / sim.entry[i] - 1.0 if held else 0.0,
                    1.0 if self.shown(i) else 0.0]

            if self.s.actions == "orders":
                resting = sim.resting(i)
                out += [float(len(resting)),
                        min(abs(order["price"] - mark) / inst.tick for order in resting)
                        if resting else 0.0]

        margin = sim.margin_equity()
        kept   = row["maintenance"]
        ratio  = 0.0 if kept == 0.0 else (1.0 if margin <= 0.0 else kept / margin)
        out += [self.returned,
                row["gross_exposure"] / equity if equity > 0.0 else 0.0,
                ratio, 1.0 - sum(weights)]

        return out


    def observe(self):
        """The flat observation, the market block's (W, N, F) row-major and then the
        account block."""
        cells  = []
        blocks = [self.market(i) for i in self.slots]
        for k in range(self.s.window):
            for block in blocks:
                cells += block[k]

        return cells + self.account()


    def state(self, before, fills, funding):
        """What a reward or a decode reads, as one environment's row of the batch's
        arrays."""
        sim    = self.sim
        equity = sim.equity()
        traded = sum(abs(sim.instruments[f["instrument"]].notional(f["size"],
                                                                   f["price"]))
                     for f in fills)
        one    = lambda x: np.array([x], dtype=np.float64)

        return types.SimpleNamespace(
            equity          = one(equity),
            previous_equity = one(before),
            log_return      = one(self.returned),
            drawdown        = one(1.0 - equity / self.peak),
            leverage        = one(sim.ledger[-1]["gross_exposure"] / equity
                                  if equity > 0.0 else 0.0),
            turnover        = one(traded / before),
            fees            = one(sum(f["fee"] for f in fills) / before),
            funding         = one(funding / before),
            liquidated      = np.array([sim.liquidations > self.liquidations]),
            weights         = np.array([[self.weight(i, equity) for i in self.slots]]),
            positions       = np.array([[sim.position[i] for i in self.slots]]),
        )


    def decode(self, action, decode=None, state=None):
        """Actions: the weights each shown slot is targeted to, or for orders the
        limit each places; an unshown slot is left out."""
        n, lever = len(self.slots), self.s.max_leverage
        shown = [self.shown(i) for i in self.slots]
        spot  = [self.sim.instruments[i].kind == "spot" for i in self.slots]

        if decode is not None:
            weights = np.asarray(decode(np.asarray([action]), state),
                                 dtype=np.float64)[0]
            return {i: float(w) for i, w, on in zip(self.slots, weights, shown)
                    if on and not math.isnan(w)}

        if self.s.actions == "box":
            a = [min(max(float(x), 0.0 if low else -1.0), 1.0)
                 for x, low in zip(action, spot)]
            total = max(1.0, sum(abs(x) for x, on in zip(a, shown) if on))
            return {i: lever * x / total
                    for i, x, on in zip(self.slots, a, shown) if on}

        if self.s.actions == "simplex":
            a     = [min(max(float(x), 0.0), 1.0) for x in action]
            total = sum(x for x, on in zip(a[:n], shown) if on) + a[n]
            return {i: x / total if total > 0.0 else 0.0
                    for i, x, on in zip(self.slots, a[:n], shown) if on}

        if self.s.actions == "discrete":
            return {i: (int(k) if low else int(k) - 1) * lever / n
                    for i, k, low, on in zip(self.slots, action, spot, shown) if on}

        raise ValueError(self.s.actions)


    def place(self, action):
        """Actions, orders: each shown slot's limit, replacing the one the last step
        left resting."""
        sim    = self.sim
        equity = sim.equity()

        for i, (size, offset), on in zip(self.slots, action,
                                         [self.shown(i) for i in self.slots]):
            if not on:
                continue
            if i in self.ordered:
                sim.cancel(self.ordered.pop(i), "replaced by the next step's order")

            asked = limit(size, offset, equity, sim.mark[i], sim.instruments[i],
                          self.s.max_leverage, self.s.max_offset)
            if asked is not None:
                side, amount, price = asked
                self.ordered[i] = sim.order(i, side, amount, limit=price)


    def step(self, action, decode=None, reward=None):
        """One step: the decision at the current close, the next bar, and its
        observation, reward and flags (Specification, Reward and termination)."""
        sim    = self.sim
        before = sim.equity()
        self.liquidations = sim.liquidations

        if self.s.actions == "orders":
            self.place(action)
        else:
            weights = self.decode(action, decode, self.last)
            if weights:
                sim.target(weights, tolerance=self.s.tolerance)

        sim.step()
        self.steps += 1

        fills         = [fill for fill in sim.fills if fill["bar"] == sim.t]
        funding       = sim.ledger[-1]["funding"]
        equity        = sim.equity()
        returned      = logged(equity, before, self.floor)
        self.returned = returned
        self.peak     = max(self.peak, equity)
        self.last     = self.state(before, fills, funding)

        if reward is not None:
            got = float(np.asarray(reward(self.last))[0])
        elif self.s.reward == "sharpe":
            a, b = self.moments
            da, db = returned - a, returned * returned - b
            spread = b - a * a
            got    = ((b * da - a * db / 2.0) / spread ** 1.5 if spread > 0.0
                      else 0.0)
            eta          = 2.0 / (self.s.window + 1)
            self.moments = [a + eta * da, b + eta * db]
        else:
            got = returned

        terminated = sim.liquidations > self.liquidations or equity <= self.floor
        truncated  = (not terminated
                      and (sim.t == len(sim.times) - 1
                           or self.s.episode_length is not None
                           and self.steps == self.s.episode_length))
        self.ended = terminated or truncated

        return self.observe(), got, terminated, truncated
