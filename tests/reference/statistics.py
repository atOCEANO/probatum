"""probatum's reference statistics, written from the Specification's Inference section.

Like the reference simulator, it exists to disagree with `probatum.stats`, so it is
written from the rules as the page states them and never from the module. Each
function computes one rule with plain loops over lists and names the section that
states it. It draws nothing: where a statistic resamples, it takes the rows a resample
drew, so a test can hand it the module's own draws and compare what the two compute
from them.
"""

import itertools
import math
import statistics

NORMAL   = statistics.NormalDist()
NUDGE    = 1e-9      # Intervals of the returns: a rank this near a whole number is it
CEILING  = 1e12      # what a CAGR saturates at, as the record's does
FLOOR    = 60        # Samples with no interval: the round trips a trade interval needs
SPAN     = 500       # Samples with no interval: the returns a bootstrap of them needs
BLOCKS   = 100       # Samples with no interval: the mean blocks those returns hold
ROUNDING = 1e-20     # Two Sharpes on the same bars: a variance below it counts as zero
MOST     = 100       # The effective number of trials: the most clusters a start tries
LLOYD    = 300       # The effective number of trials: Lloyd's iterations at most
PARTS    = 16        # The probability of backtest overfitting: the blocks of a run
# The deflated Sharpe: the Euler-Mascheroni constant
EULER = 0.5772156649015329
# Intervals of the returns: the statistics read on the growth a bar, and every one
# studentized on a pivot
GROWTH      = ("total_return_pct", "net_profit_pct", "cagr_pct")
STUDENTIZED = (*GROWTH, "volatility_pct", "sortino", "conditional_value_at_risk")

# Consistency and decay: the score past which a stretch gains, by objective
BREAK_EVEN = {
    "sharpe":           0.0,
    "sortino":          0.0,
    "calmar":           0.0,
    "total_return_pct": 0.0,
    "net_profit_pct":   0.0,
    "cagr_pct":         0.0,
    "avg_trade_pct":    0.0,
    "profit_factor":    1.0,
}
# Consistency and decay: the objectives that grow with their span
GROWS = ("total_return_pct", "net_profit_pct", "max_drawdown_pct", "num_trades",
         "num_fills", "funding_paid")


def mean(values):
    return sum(values) / len(values)


def deviation(values, ddof):
    """The deviation with `n - ddof` in its divisor."""
    centre = mean(values)
    return math.sqrt(sum((v - centre) ** 2 for v in values) / (len(values) - ddof))


def ranked(reward, risk, annualizer):
    """A reward over a risk that can be zero, ranked as the record ranks it: infinite
    for a positive reward over no risk, and zero for any other."""
    if risk > 0.0:
        return reward / risk * annualizer
    return math.inf if reward > 0.0 else 0.0


def excess(returns, periods_per_year, risk_free):
    """What a statistic reads: each return less the annual rate's share of a bar."""
    return [r - risk_free / periods_per_year for r in returns]


def per_bar(x):
    """What a statistic reads: the Sharpe of one bar, `n - 1` in its deviation."""
    return mean(x) / deviation(x, 1)


def sharpe(x, periods_per_year):
    """What a statistic reads: the record's Sharpe."""
    return ranked(mean(x), deviation(x, 1), math.sqrt(periods_per_year))


def recorded(returns, periods_per_year, risk_free):
    """The probability of backtest overfitting: the Sharpe as the record computes it, the
    excess mean over the deviation of the returns themselves, so returns that never
    vary rank at zero whatever the rate."""
    excess = mean(returns) - risk_free / periods_per_year
    return ranked(excess, deviation(returns, 1), math.sqrt(periods_per_year))


def autocorrelation(x, k):
    """What a statistic reads: the autocorrelation at lag `k`, 1 at lag 0, and 0 where
    the returns never vary or number no more than `k + 1`."""
    if k == 0:
        return 1.0

    centre = mean(x)
    below  = sum((v - centre) ** 2 for v in x)
    if len(x) <= k + 1 or below == 0.0:
        return 0.0

    return sum((x[t] - centre) * (x[t - k] - centre) for t in range(k, len(x))) / below


def effective_sample(x):
    """What a statistic reads: `n * (1 - a) / (1 + a)` at the first autocorrelation
    held to [0, 0.99], at least 2, and `n` below 4 returns."""
    n = len(x)
    if n < 4:
        return float(n)

    a = min(max(autocorrelation(x, 1), 0.0), 0.99)
    return max(2.0, n * (1.0 - a) / (1.0 + a))


def influence_at(x, mu, s):
    """The Sharpe's variance: `z_t = u_t - (S / 2) * (u_t^2 - 1)`, `u_t = (x_t - mu) / s`
    and `S = mu / s`, at a given mean and deviation."""
    shape = mu / s
    return [(v - mu) / s - shape / 2.0 * (((v - mu) / s) ** 2 - 1.0) for v in x]


def influence(x):
    """The Sharpe's variance: the influence series on the moments with `n` in every
    divisor."""
    return influence_at(x, mean(x), deviation(x, 0))


def autocovariances(z, upto):
    """The Sharpe's variance: `g_j = (1/n) * sum_{t > j} z_t * z_{t-j}` for `j` from 0
    to `upto`, an empty sum past the series."""
    n = len(z)
    return [sum(z[t] * z[t - j] for t in range(j, n)) / n for j in range(upto + 1)]


def lag(z):
    """The Sharpe's variance: Newey and West's 1994 rule for the Bartlett kernel, on a
    series whose mean is zero."""
    n  = len(z)
    p  = math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))
    g  = autocovariances(z, p)
    s0 = g[0] + 2.0 * sum(g[1:])
    s1 = 2.0 * sum(j * g[j] for j in range(1, p + 1))
    if s0 <= 0.0:
        return 0

    return min(math.floor(1.1447 * ((s1 / s0) ** 2 * n) ** (1.0 / 3.0)), n - 1)


def bartlett(g, m):
    """The Sharpe's variance: the Bartlett sum at lag `m` of the autocovariances `g`."""
    return g[0] + 2.0 * sum((1.0 - j / (m + 1.0)) * g[j] for j in range(1, m + 1))


def floored(z):
    """The Sharpe's variance: the Bartlett sum at the 1994 lag on `z`, floored at its
    lag-zero value, with that lag and whether the floor bound."""
    m      = lag(z)
    g      = autocovariances(z, m)
    summed = bartlett(g, m)
    return max(summed, g[0]), m, summed < g[0]


def sharpe_interval(x, periods_per_year, level=0.95):
    """The Sharpe's variance: the interval's bounds a year, its lag and whether the
    floor bound, or None where the variance is zero."""
    variance, m, bound = floored(influence(x))
    if variance < ROUNDING:
        return None

    error = math.sqrt(variance / (len(x) - 1))
    z     = NORMAL.inv_cdf((1.0 + level) / 2.0)
    root  = math.sqrt(periods_per_year)
    return (per_bar(x) - z * error) * root, (per_bar(x) + z * error) * root, m, bound


def lag_zero(shape, k3, k4):
    """The Sharpe's variance: `g_0 = 1 - S * k3 + (S^2 / 4) * (k4 - 1)`."""
    return 1.0 - shape * k3 + shape * shape / 4.0 * (k4 - 1.0)


def probability(shape, benchmark, variance, n):
    """The probabilistic Sharpe and the minimum track record: `Phi((S - B) * sqrt(n - 1)
    / sqrt(V))` for a per-bar Sharpe and benchmark."""
    return NORMAL.cdf((shape - benchmark) * math.sqrt(n - 1) / math.sqrt(variance))


def track_record(shape, benchmark, variance, confidence):
    """The probabilistic Sharpe and the minimum track record: `1 + V * (Phi^-1(c) / (S -
    B))^2` returns, or None for a Sharpe not above the benchmark."""
    if shape <= benchmark:
        return None
    return 1.0 + variance * (NORMAL.inv_cdf(confidence) / (shape - benchmark)) ** 2


def probabilistic_sharpe(x, periods_per_year, benchmark=0.0):
    """The probabilistic Sharpe and the minimum track record: the probability against
    an annual Sharpe, or None where the variance is zero."""
    variance = floored(influence(x))[0]
    if variance < ROUNDING:
        return None

    return probability(per_bar(x), benchmark / math.sqrt(periods_per_year), variance,
                       len(x))


def min_track_record(x, periods_per_year, benchmark=0.0, confidence=0.95):
    """The probabilistic Sharpe and the minimum track record: the returns it needs, or
    None where the variance is zero."""
    variance = floored(influence(x))[0]
    if variance < ROUNDING:
        return None

    return track_record(per_bar(x), benchmark / math.sqrt(periods_per_year), variance,
                        confidence)


def long_run_sharpe(x, periods_per_year):
    """The long-run Sharpe: the mean over the Bartlett long-run deviation of the
    returns at the 1994 lag on them, not floored."""
    mu       = mean(x)
    centred  = [v - mu for v in x]
    m        = lag(centred)
    variance = bartlett(autocovariances(centred, m), m)
    return math.sqrt(periods_per_year) * mu / math.sqrt(variance)


def correlation(a, b):
    """Intervals of the returns: Pearson's correlation, each side centred on its own
    mean, None where a side never varies."""
    ca, cb = mean(a), mean(b)
    top    = sum((p - ca) * (q - cb) for p, q in zip(a, b))
    bottom = sum((p - ca) ** 2 for p in a) * sum((q - cb) ** 2 for q in b)
    return None if bottom == 0.0 else top / math.sqrt(bottom)


def covariance(a, b):
    """Intervals of the returns: the covariance, each side centred on its own mean."""
    ca, cb = mean(a), mean(b)
    return sum((p - ca) * (q - cb) for p, q in zip(a, b)) / (len(a) - 1)


def stationary(ratio, count):
    """Intervals of the returns: `abs(G / g)^(2/3) * N^(1/3)`, before it is held."""
    return abs(ratio) ** (2.0 / 3.0) * count ** (1.0 / 3.0)


def block_lengths(y):
    """Intervals of the returns: the stationary and the circular lengths that Politis
    and White's rule gives `y`, as Patton's code computes them, with the page's
    departures."""
    count = len(y)
    k     = 5
    m_max = math.ceil(math.sqrt(count)) + k
    b_max = math.ceil(min(3.0 * math.sqrt(count), count / 3.0))
    if count < m_max + 3 or all(v == y[0] for v in y):
        return 1.0, 1

    h   = 2.0 * math.sqrt(math.log10(count) / count)
    rho = [None] + [correlation(y[m_max:], y[m_max - j:count - j])
                    for j in range(1, m_max + 1)]
    if any(r is None for r in rho[1:]):
        return 1.0, 1

    m_hat = next((j for j in range(1, m_max - k + 2)
                  if all(abs(rho[i]) < h for i in range(j, j + k))), None)
    if m_hat is None:
        m_hat = max(j for j in range(1, m_max + 1) if abs(rho[j]) > h)

    big_m = min(2 * m_hat, m_max)
    r     = [covariance(y[big_m:], y[big_m - j:count - j]) for j in range(big_m + 1)]
    w     = [None] + [1.0 if j / big_m < 0.5 else 2.0 * (1.0 - j / big_m)
                      for j in range(1, big_m + 1)]
    big_g = 2.0 * sum(w[j] * j * r[j] for j in range(1, big_m + 1))
    g     = r[0] + 2.0 * sum(w[j] * r[j] for j in range(1, big_m + 1))
    raw   = math.inf if g == 0.0 else stationary(big_g / g, count)
    return (max(1.0, min(raw, b_max)),
            max(1, math.ceil(min(1.5 ** (1.0 / 3.0) * raw, b_max))))


def tail(r):
    """Intervals of the returns: `k = ceil(n / 20)`, the count of the lowest returns
    that value at risk reads."""
    return (len(r) + 19) // 20


def return_block(r):
    """Intervals of the returns: the largest stationary length of the returns, their
    squared deviations from their mean and the indicator of the lowest."""
    centre = mean(r)
    cutoff = sorted(r)[tail(r) - 1]
    return max(block_lengths(series)[0] for series in (
        r, [(v - centre) ** 2 for v in r], [1.0 if v <= cutoff else 0.0 for v in r]))


def compounded(r):
    """Intervals of the returns: the equity a resample ends on from 1, a return after
    it reaches zero moving it no further."""
    equity = 1.0
    for v in r:
        if equity > 0.0:
            equity *= 1.0 + v
    return equity


def total_return(r):
    return (compounded(r) - 1.0) * 100.0


def cagr(r, periods_per_year):
    equity = compounded(r)
    if equity <= 0.0:
        return -100.0

    return min(equity ** (periods_per_year / len(r)) - 1.0, CEILING) * 100.0


def volatility(r, periods_per_year):
    return deviation(r, 1) * math.sqrt(periods_per_year) * 100.0


def sortino(x, periods_per_year):
    """The excess returns' mean over the population deviation of those below zero."""
    downside = math.sqrt(sum(min(v, 0.0) ** 2 for v in x) / len(x))
    return ranked(mean(x), downside, math.sqrt(periods_per_year))


def max_drawdown(r):
    equity = peak = 1.0
    worst = 0.0

    for v in r:
        if equity > 0.0:
            equity *= 1.0 + v
        peak = max(peak, equity)
        if peak > 0.0:
            worst = max(worst, min((peak - equity) / peak, 1.0))

    return worst * 100.0


def calmar(r, periods_per_year):
    """A losing run's growth scaled by its drawdown, any other's divided by it."""
    growth = cagr(r, periods_per_year) / 100.0
    fall   = max_drawdown(r) / 100.0
    return growth * fall if growth < 0.0 else ranked(growth, fall, 1.0)


def value_at_risk(r):
    """Intervals of the returns: the loss of the `k`-th lowest return, in percent."""
    return -sorted(r)[tail(r) - 1] * 100.0


def conditional_value_at_risk(r):
    """Intervals of the returns: the mean loss of the `k` lowest returns, in percent."""
    return -mean(sorted(r)[:tail(r)]) * 100.0


def ordered(values, q):
    """Intervals of the returns: the value of rank `ceil(B * q)` among the `B`, a rank
    within `1e-9` of a whole number being that number, the first at least."""
    rank = max(1, math.ceil(len(values) * q - NUDGE))
    return sorted(values)[rank - 1]


def percentile(values, level):
    """Intervals of the returns: the percentile interval's bounds."""
    return ordered(values, (1.0 - level) / 2.0), ordered(values, (1.0 + level) / 2.0)


def long_run(z, m):
    """Intervals of the returns: the Bartlett sum of `z` at lag `m`, floored at its
    lag-zero value."""
    g = autocovariances(z, m)
    return max(bartlett(g, m), g[0])


def pivot(name, r, x):
    """Intervals of the returns: what a studentized interval of `name` is read on and
    its influence series, with `n` in every divisor, or no series where the returns
    leave it nothing to divide by."""
    n = len(r)

    if name in GROWTH:
        logs = [math.log1p(v) for v in r]
        g    = mean(logs)
        return g, [v - g for v in logs]

    if name == "volatility_pct":
        centre = mean(r)
        s2     = sum((v - centre) ** 2 for v in r) / n
        if s2 == 0.0:
            return -math.inf, None
        return (0.5 * math.log(s2 * n / (n - 1)),
                [((v - centre) ** 2 - s2) / (2.0 * s2) for v in r])

    if name == "sortino":
        mu = mean(x)
        d2 = sum(min(v, 0.0) ** 2 for v in x) / n
        if d2 == 0.0:
            return ranked(mu, 0.0, 1.0), None
        d = math.sqrt(d2)
        return mu / d, [(v - mu) / d - mu / (2.0 * d ** 3) * (min(v, 0.0) ** 2 - d2)
                        for v in x]

    k      = tail(r)
    lowest = sorted(r)[:k]
    q, below = lowest[-1], mean(lowest)
    return below, [q + ((v - q) * n / k if v <= q else 0.0) - below for v in r]


def ratio(move, error):
    """Intervals of the returns: a resample's studentized statistic, infinite in the
    sign of a move with no error to measure it, and 0 where it did not move."""
    if error is not None and 0.0 < error < math.inf:
        return move / error
    return 0.0 if move == 0.0 else math.copysign(math.inf, move)


def grown(value):
    """`exp(value) - 1`, infinite past what a float holds."""
    try:
        return math.expm1(value)
    except OverflowError:
        return math.inf


def unpivoted(name, theta, n, periods_per_year):
    """Intervals of the returns: a bound of `name` from its pivot's bound `theta`."""
    if name in ("total_return_pct", "net_profit_pct"):
        return grown(n * theta) * 100.0

    if name == "cagr_pct":
        return min(grown(periods_per_year * theta), CEILING) * 100.0

    if name == "volatility_pct":
        return math.exp(theta) * math.sqrt(periods_per_year) * 100.0

    if name == "sortino":
        return theta * math.sqrt(periods_per_year)

    return -theta * 100.0


def studentized_returns(name, r, x, samples, periods_per_year, level=0.95):
    """Intervals of the returns: the studentized interval of `name` from the resampled
    rows `samples`, each resample's error at the sample's own lag."""
    n = len(r)
    theta, psi = pivot(name, r, x)
    m     = lag(psi)
    error = math.sqrt(long_run(psi, m) / (n - 1))
    t     = []

    for row in samples:
        moved, spread = pivot(name, [r[i] for i in row], [x[i] for i in row])
        own = None if spread is None else math.sqrt(long_run(spread, m) / (n - 1))
        t.append(ratio(moved - theta, own))

    low, high = percentile(t, level)
    return tuple(sorted((unpivoted(name, theta - high * error, n, periods_per_year),
                         unpivoted(name, theta - low * error, n, periods_per_year))))


def win_rate(pnl):
    return sum(1 for v in pnl if v > 0.0) / len(pnl)


def profit_factor(pnl):
    gain = sum(v for v in pnl if v > 0.0)
    loss = -sum(v for v in pnl if v < 0.0)
    return ranked(gain, loss, 1.0)


def average_trade(pnl, opening):
    return mean(pnl) / opening * 100.0


def corrected(values, estimate, jackknife, level):
    """Intervals of the round trips: the bias-corrected and accelerated bounds of
    resampled `values` around `estimate`, its acceleration from the `jackknife`."""
    below = (sum(1 for v in values if v < estimate)
             + 0.5 * sum(1 for v in values if v == estimate)) / len(values)
    z0     = NORMAL.inv_cdf(below)
    centre = mean(jackknife)
    spread = sum((centre - p) ** 2 for p in jackknife)
    a      = sum((centre - p) ** 3 for p in jackknife) / (6.0 * spread ** 1.5)
    bounds = []
    for q in ((1.0 - level) / 2.0, (1.0 + level) / 2.0):
        z = NORMAL.inv_cdf(q)
        bounds.append(ordered(values, NORMAL.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))))

    return tuple(bounds)


def profit_factor_interval(pnl, samples, level=0.95):
    """Intervals of the round trips: the profit factor's interval from the resampled
    trips `samples`."""
    jackknife = [profit_factor(pnl[:i] + pnl[i + 1:]) for i in range(len(pnl))]
    return corrected([profit_factor(s) for s in samples], profit_factor(pnl), jackknife,
                     level)


def average_trade_interval(pnl, samples, opening, level=0.95):
    """Intervals of the round trips: the average trade's studentized interval from the
    resampled trips `samples`, in percent of the opening balance."""
    k = len(pnl)
    m, s, root = mean(pnl), deviation(pnl, 1), math.sqrt(k)
    t = []

    for sample in samples:
        ms, ss = mean(sample), deviation(sample, 1)
        if ss > 0.0:
            t.append((ms - m) / (ss / root))
        else:
            t.append(0.0 if ms == m else math.copysign(math.inf, ms - m))

    low, high = percentile(t, level)
    return (m - high * s / root) / opening * 100.0, (m - low * s / root) / opening * 100.0


def supports_returns(x):
    """Samples with no interval: 4 returns at least, and returns that vary."""
    return len(x) >= 4 and any(v != x[0] for v in x)


def supports_bootstrap(r, x, name):
    """Samples with no interval: what a bootstrap of the returns `r`, whose excess
    returns are `x`, needs of them for `name`: no return of -100% or worse for a
    statistic read on growth, a tail that varies for conditional value at risk, and a
    standard error above zero for a studentized statistic."""
    if name in GROWTH and min(r) <= -1.0:
        return False

    if len(r) < SPAN or len(r) < BLOCKS * return_block(r):
        return False

    if name == "conditional_value_at_risk" and len(set(sorted(r)[:tail(r)])) == 1:
        return False

    if name in STUDENTIZED:
        psi = pivot(name, r, x)[1]
        return psi is not None and long_run(psi, lag(psi)) > 0.0

    return True


def supports_trips(pnl, name):
    """Samples with no interval: what a trade statistic's interval needs of the closed
    round trips."""
    if len(pnl) < FLOOR:
        return False

    won  = sum(1 for v in pnl if v > 0.0)
    lost = sum(1 for v in pnl if v < 0.0)

    if name == "win_rate":
        return 0 < won < len(pnl)

    if name == "profit_factor":
        return won > 0 and lost > 1

    return any(v != pnl[0] for v in pnl)


def difference(xa, xb, periods_per_year, level=0.95):
    """Two Sharpes on the same bars: the delta method's difference a year, its standard
    error a year, its p-value, its interval's bounds, its lag and whether its floor
    bound, or None for two series whose influence is the same on every bar."""
    variance, m, bound = floored([a - b for a, b in zip(influence(xa), influence(xb))])
    if variance < ROUNDING:
        return None

    d     = per_bar(xa) - per_bar(xb)
    error = math.sqrt(variance / (len(xa) - 1))
    q     = NORMAL.inv_cdf((1.0 + level) / 2.0)
    root  = math.sqrt(periods_per_year)
    return (d * root, error * root, 2.0 * NORMAL.cdf(-abs(d) / error),
            ((d - q * error) * root, (d + q * error) * root), m, bound)


def circular(count, length, starts):
    """Two Sharpes on the same bars: the rows a circular block resample takes from its
    blocks' starting rows."""
    return [(start + i) % count for start in starts for i in range(length)][:count]


def natural_variance(z, length):
    """Two Sharpes on the same bars: `V*` over the resample's whole blocks."""
    q = len(z) // length
    return sum(sum(z[i * length:(i + 1) * length]) ** 2 / length for i in range(q)) / q


def bootstrap_p(studentized, statistic):
    """Two Sharpes on the same bars: `(#{d* >= abs(D) / se} + 1) / (B + 1)`."""
    return (sum(1 for d in studentized if d >= statistic) + 1) / (len(studentized) + 1)


def studentized_difference(xa, xb, periods_per_year, resamples, level=0.95):
    """Two Sharpes on the same bars: Ledoit and Wolf's p-value and interval a year from
    each resample's blocks' starting rows, or None for two series whose influence is
    the same on every bar."""
    count = len(xa)
    whole = [a - b for a, b in zip(influence(xa), influence(xb))]
    if floored(whole)[0] < ROUNDING:
        return None

    length = block_lengths(whole)[1]
    d      = per_bar(xa) - per_bar(xb)
    error  = math.sqrt(floored(whole)[0] / (count - 1))
    found  = []

    for starts in resamples:
        rows = circular(count, length, starts)
        a    = [xa[i] for i in rows]
        b    = [xb[i] for i in rows]
        if deviation(a, 0) == 0.0 or deviation(b, 0) == 0.0:
            found.append(math.inf)
            continue

        z      = [p - q for p, q in zip(influence(a), influence(b))]
        spread = math.sqrt(natural_variance(z, length) / (count - 1))
        found.append(math.inf if spread == 0.0 else abs(per_bar(a) - per_bar(b) - d) / spread)

    half = ordered(found, level) * error
    root = math.sqrt(periods_per_year)
    return (bootstrap_p(found, abs(d) / error), ((d - half) * root, (d + half) * root),
            length)


def product(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]


def inverse(a):
    """Gauss and Jordan's inverse with partial pivoting, for the few columns a
    regression here has."""
    size = len(a)
    work = [list(row) + [1.0 if i == j else 0.0 for j in range(size)]
            for i, row in enumerate(a)]

    for col in range(size):
        pivot = max(range(col, size), key=lambda i: abs(work[i][col]))
        work[col], work[pivot] = work[pivot], work[col]
        lead = work[col][col]
        work[col] = [v / lead for v in work[col]]
        for i in range(size):
            if i != col:
                factor = work[i][col]
                work[i] = [v - factor * p for v, p in zip(work[i], work[col])]

    return [row[size:] for row in work]


def least_squares(rows, y):
    """Alpha and beta: the coefficients of `y` on `rows` by least squares, and the
    residuals."""
    columns = list(zip(*rows))
    gram    = [[sum(p * q for p, q in zip(a, b)) for b in columns] for a in columns]
    moved   = [[sum(p * q for p, q in zip(a, y))] for a in columns]
    coef    = [c[0] for c in product(inverse(gram), moved)]
    fitted  = [sum(c * v for c, v in zip(coef, row)) for row in rows]
    return coef, [p - q for p, q in zip(y, fitted)]


def newey_west(rows, e, m):
    """Alpha and beta: `(X'X)^-1 * W * (X'X)^-1` with Bartlett weights at lag `m` and no
    small-sample factor."""
    size    = len(rows[0])
    columns = list(zip(*rows))
    gram    = [[sum(p * q for p, q in zip(a, b)) for b in columns] for a in columns]
    w       = [[0.0] * size for _ in range(size)]

    for t, row in enumerate(rows):
        for i in range(size):
            for j in range(size):
                w[i][j] += e[t] * e[t] * row[i] * row[j]

    for k in range(1, m + 1):
        weight = 1.0 - k / (m + 1.0)
        for t in range(k, len(rows)):
            joint = weight * e[t] * e[t - k]
            for i in range(size):
                for j in range(size):
                    w[i][j] += joint * (rows[t][i] * rows[t - k][j]
                                        + rows[t - k][i] * rows[t][j])

    bread = inverse(gram)
    return product(product(bread, w), bread)


def alpha_beta(x, y, periods_per_year, given=None, level=0.95):
    """Alpha and beta: alpha a year and its standard error, beta and its standard
    error, the lag, the tracking error and the information ratio, for excess returns
    `x` on a benchmark's `y`."""
    rows = [[1.0, v] for v in y]
    coef, e = least_squares(rows, x)

    m                 = given if given is not None else lag([v * r for v, r in zip(y, e)])
    covariance_matrix = newey_west(rows, e, m)
    active            = [p - q for p, q in zip(x, y)]
    root              = math.sqrt(periods_per_year)
    return dict(alpha=coef[0] * periods_per_year,
                alpha_error=math.sqrt(covariance_matrix[0][0]) * periods_per_year,
                beta=coef[1], beta_error=math.sqrt(covariance_matrix[1][1]), lag=m,
                tracking_error=deviation(active, 1) * root,
                information_ratio=mean(active) / deviation(active, 1) * root)


def trade_path(pnl, opened_on, order):
    """The order of trades: the drawdown at trade closes, in percent, and the longest
    losing run of the trips in `order`, each returning its pnl on the equity it opened
    on."""
    equity = peak = 1.0
    worst = 0.0
    run = longest = 0

    for i in order:
        if equity > 0.0:
            equity = max(0.0, equity * (1.0 + pnl[i] / opened_on[i]))
        peak    = max(peak, equity)
        worst   = max(worst, min((peak - equity) / peak, 1.0))
        run     = run + 1 if pnl[i] < 0.0 else 0
        longest = max(longest, run)

    return worst * 100.0, longest


def deeper(falls, fall):
    """The order of trades: the share of the permutations' drawdowns `falls` deeper than
    the run's `fall` by more than a part in a trillion."""
    return sum(1 for f in falls if f > fall * (1.0 + 1e-12)) / len(falls)


def stretches(first, last, count):
    """Consistency and decay: rows `first` to `last` divided as PurgedKFold divides
    bars, the first `rows mod count` one longer."""
    rows = last - first + 1
    size, extra = divmod(rows, count)
    spans = []
    start = first
    for i in range(count):
        end = start + size + (1 if i < extra else 0) - 1
        spans.append((start, end))
        start = end + 1

    return spans


def gained(scores, objective, direction):
    """Consistency and decay: how many scores are past the objective's break-even in
    the search's direction, or None for an objective that has none."""
    if objective not in BREAK_EVEN:
        return None

    even = BREAK_EVEN[objective]
    return sum(1 for s in scores if (s > even if direction == "maximize" else s < even))


def decay(fitted, earned, direction):
    """Consistency and decay: each gap, signed so a positive one is worse out of
    sample, and their mean."""
    gaps = [f - e if direction == "maximize" else e - f for f, e in zip(fitted, earned)]
    return gaps, mean(gaps)


def expected_maximum(trials):
    """The deflated Sharpe: `f(N) = (1 - gamma) * Phi^-1(1 - 1/N) + gamma * Phi^-1(1 -
    1/(N * e))`, the expected best of `N` standard normal draws."""
    return ((1.0 - EULER) * NORMAL.inv_cdf(1.0 - 1.0 / trials)
            + EULER * NORMAL.inv_cdf(1.0 - 1.0 / (trials * math.e)))


def deflation_threshold(spread, trials):
    """The deflated Sharpe: the annual Sharpe the best of `trials` reaches by luck when
    their annual Sharpes spread `spread`."""
    return spread * expected_maximum(trials)


def deflated_sharpe(x, periods_per_year, spread, trials):
    """The deflated Sharpe: the winner's probabilistic Sharpe against the threshold of
    `trials` whose annual Sharpes spread `spread`, 0 at one trial."""
    threshold = 0.0 if trials == 1 else deflation_threshold(spread, trials)
    return probabilistic_sharpe(x, periods_per_year, threshold)


def min_backtest_length(trials, sharpe):
    """The minimum backtest length: `(f(N) / S)^2` years at an annual Sharpe `S`."""
    return (expected_maximum(trials) / sharpe) ** 2


def distances(rows):
    """The effective number of trials: each trial's row of `D = sqrt((1 - rho) / 2)`,
    and the Euclidean distances between those rows."""
    count = len(rows)
    d     = [[0.0 if i == j else
              math.sqrt(max(0.0, (1.0 - correlation(rows[i], rows[j])) / 2.0))
              for j in range(count)] for i in range(count)]

    far = [[math.sqrt(sum((d[i][k] - d[j][k]) ** 2 for k in range(count)))
            for j in range(count)] for i in range(count)]
    return d, far


def silhouettes(far, labels):
    """The effective number of trials: `s_i = (b_i - a_i) / max(a_i, b_i)`, 0 for a
    trial alone in its cluster."""
    out = []
    for i, own in enumerate(labels):
        mates = [j for j, label in enumerate(labels) if label == own and j != i]
        if not mates:
            out.append(0.0)
            continue

        a = mean([far[i][j] for j in mates])
        b = min(mean([far[i][j] for j, label in enumerate(labels) if label == other])
                for other in set(labels) if other != own)
        out.append((b - a) / max(a, b) if max(a, b) > 0.0 else 0.0)

    return out


def quality(values):
    """The effective number of trials: the mean over the deviation, `n` in its
    divisor, infinite at no deviation and a mean above 0, and 0 at no deviation
    otherwise."""
    spread = deviation(values, 0)
    if spread == 0.0:
        return math.inf if mean(values) > 0.0 else 0.0

    return mean(values) / spread


def sequence(squared, most, draws):
    """The effective number of trials: one greedy k-means++ sequence of `most` trials at
    `squared` distances, read off `draws`, the stream's draws in order: the first trial,
    then each step's `2 + floor(ln most)` even draws `u`, each taking the first trial
    whose running sum of squared distances from the sequence so far is above `u` times
    their total, or the trials drawn evenly where every distance is 0; the step keeps
    the one leaving the least sum of every trial's squared distance to its nearest, the
    first of equal ones."""
    count = len(squared)
    order = [int(next(draws))]
    near  = list(squared[order[0]])

    for _ in range(1, most):
        running = list(itertools.accumulate(near))
        given   = [float(v) for v in next(draws)]
        if running[-1] > 0.0:
            picks = [next((i for i, v in enumerate(running) if v > u * running[-1]),
                          count - 1) for u in given]
        else:
            picks = [int(v) for v in given]

        sums = [sum(min(a, b) for a, b in zip(near, squared[p])) for p in picks]
        best = picks[sums.index(min(sums))]
        order.append(best)
        near = [min(a, b) for a, b in zip(near, squared[best])]

    return order


def lloyd(points, centres):
    """The effective number of trials: Lloyd's iterations from the trials `centres`,
    each trial to its nearest centre, the lowest-numbered on ties, each cluster left
    empty taking in turn the trial farthest from its own centre, and one emptied by
    that keeping its centre, until no trial moves."""
    count, k = len(points), len(centres)
    middle = [list(points[c]) for c in centres]
    labels = None

    for _ in range(LLOYD):
        near = [[sum((p - c) ** 2 for p, c in zip(points[i], middle[j]))
                 for j in range(k)] for i in range(count)]
        new = [min(range(k), key=lambda j: (near[i][j], j)) for i in range(count)]
        if new == labels:
            break

        labels = new
        moved  = set()
        for j in range(k):
            if j in labels:
                continue
            worst = max((i for i in range(count) if i not in moved),
                        key=lambda i: (near[i][labels[i]], -i))
            labels[worst] = j
            moved.add(worst)

        for j in range(k):
            members = [points[i] for i in range(count) if labels[i] == j]
            if members:
                middle[j] = [mean(column) for column in zip(*members)]

    return labels


def numbered(labels):
    """The effective number of trials: the clusters numbered by their first trial."""
    order = {}
    for label in labels:
        order.setdefault(label, len(order))
    return [order[label] for label in labels]


def base_clustering(points, far, sequences):
    """The effective number of trials: the clustering of the highest quality over each
    start's sequence and each `k`, the first on ties."""
    best = None
    for sequence in sequences:
        for k in range(2, len(sequence) + 1):
            labels = numbered(lloyd(points, sequence[:k]))
            score  = quality(silhouettes(far, labels))
            if best is None or score > best[0]:
                best = (score, labels)

    return best[1]


def onc(rows, draw):
    """The effective number of trials: the base clustering and its refinement,
    `draw(far, k_max)` giving the starts' sequences the module drew for each
    clustering."""
    count = len(rows)
    if count < 3:
        return list(range(count))

    points, far = distances(rows)
    labels = base_clustering(points, far, draw(far, min(count - 1, MOST)))
    scores = silhouettes(far, labels)
    groups = {label: [i for i in range(count) if labels[i] == label]
              for label in set(labels)}
    each   = {label: quality([scores[i] for i in members])
              for label, members in groups.items() if len(members) > 1}
    if not each:
        return labels

    middle = mean(list(each.values()))
    redo   = sorted(label for label, value in each.items() if value < middle)
    if len(redo) <= 2:
        return labels

    items  = [i for i in range(count) if labels[i] in redo]
    inner  = onc([rows[i] for i in items], draw)
    merged = list(labels)
    for i, label in zip(items, inner):
        merged[i] = ("new", label)

    merged  = numbered(merged)
    renewed = silhouettes(far, merged)
    after   = [quality([renewed[i] for i in range(count) if merged[i] == label])
               for label in sorted({merged[i] for i in items})
               if sum(1 for i in range(count) if merged[i] == label) > 1]
    before  = [each[label] for label in redo]
    if after and mean(after) > mean(before):
        return merged
    return labels


def combined(rows):
    """The effective number of trials: a cluster's series at minimum variance under one
    correlation, or inverse variance where that correlation is out of its range."""
    if len(rows) == 1:
        return list(rows[0])

    m    = len(rows)
    sd   = [deviation(row, 1) for row in rows]
    rbar = mean([correlation(rows[i], rows[j])
                 for i in range(m) for j in range(i + 1, m)])

    if -1.0 / (m - 1) < rbar < 1.0:
        inverse = sum(1.0 / s for s in sd)
        raw     = [1.0 / s ** 2 - rbar * inverse / ((1.0 + (m - 1) * rbar) * s)
                   for s in sd]
    else:
        raw = [1.0 / s ** 2 for s in sd]

    total = sum(raw)
    return [sum(raw[i] / total * rows[i][t] for i in range(m))
            for t in range(len(rows[0]))]


def effective_trials(x, periods_per_year, draw):
    """The effective number of trials: `K`, and the sample deviation of the K series'
    annual Sharpes, None at one cluster, from the excess returns `x` of the trials."""
    labels = onc(x, draw)
    series = [combined([x[i] for i in range(len(x)) if labels[i] == label])
              for label in sorted(set(labels))]
    if len(series) == 1:
        return 1, None

    return len(series), deviation([sharpe(s, periods_per_year) for s in series], 1)


def overfitting(rows, statistic, maximize=True, blocks=PARTS):
    """The probability of backtest overfitting: the share of halvings whose in-sample
    winner ranks below the median out of sample, and the share whose winner loses out
    of sample, tied winners weighted alike, each half's statistic computed on its joined
    returns."""
    size  = len(rows[0]) // blocks
    drop  = len(rows[0]) - size * blocks
    parts = [[row[drop + b * size:drop + (b + 1) * size] for b in range(blocks)]
             for row in rows]

    chosen = list(itertools.combinations(range(blocks), blocks // 2))
    over = loss = 0.0
    for picked in chosen:
        left    = [b for b in range(blocks) if b not in picked]
        inside  = [statistic([v for b in picked for v in trial[b]]) for trial in parts]
        outside = [statistic([v for b in left for v in trial[b]]) for trial in parts]
        best    = max(inside) if maximize else min(inside)
        winners = [k for k, v in enumerate(inside) if v == best]
        for k in winners:
            worse = sum(1 for v in outside
                        if (v < outside[k] if maximize else v > outside[k]))
            equal = sum(1 for v in outside if v == outside[k])
            omega = (worse + (equal + 1) / 2.0) / (len(rows) + 1)
            over += (math.log(omega / (1.0 - omega)) < 0.0) / len(winners)
            loss += (outside[k] < 0.0) / len(winners)

    return over / len(chosen), loss / len(chosen)


def relative(rows, benchmark):
    """The Reality Check and SPA: `f_k,t = ln(1 + r_k,t) - ln(1 + b_t)`."""
    return [[math.log1p(r) - math.log1p(b) for r, b in zip(row, benchmark)]
            for row in rows]


def kernel_variance(f, q):
    """The Reality Check and SPA: `g_0 + 2 * sum_i kappa(n, i) * g_i`, the stationary
    bootstrap's variance of `sqrt(n)` times a resample's mean."""
    n, centre = len(f), mean(f)
    g = [sum((f[t] - centre) * (f[t + i] - centre) for t in range(n - i)) / n
         for i in range(n)]
    kappa = [(n - i) / n * (1.0 - q) ** i + i / n * (1.0 - q) ** (n - i)
             for i in range(n)]
    return g[0] + 2.0 * sum(kappa[i] * g[i] for i in range(1, n))


def mean_block(f):
    """The Reality Check and SPA: the median of the stationary lengths of the trials
    whose relative series varies, 1 where none does."""
    lengths = [block_lengths(row)[0] for row in f if any(v != row[0] for v in row)]
    return statistics.median(lengths) if lengths else 1.0


def reality_check(f, drawn, q):
    """The Reality Check and SPA: White's p-value, and Hansen's consistent SPA with each
    resample studentized on its own variance, over the resamples `drawn`."""
    n     = len(f[0])
    fbar  = [mean(row) for row in f]
    big_v = max(math.sqrt(n) * v for v in fbar)
    white = sum(1 for rows in drawn
                if max(math.sqrt(n) * (mean([row[i] for i in rows]) - fbar[k])
                       for k, row in enumerate(f)) > big_v) / len(drawn)
    if any(all(v == row[0] for v in row) and row[0] > 0.0 for row in f):
        return white, 0.0

    live  = [k for k, row in enumerate(f) if any(v != row[0] for v in row)]
    omega = {k: math.sqrt(kernel_variance(f[k], q)) for k in live}
    t     = {k: math.sqrt(n) * fbar[k] / omega[k] for k in live}
    big_t = max([0.0] + list(t.values()))
    if big_t == 0.0:
        return white, 1.0

    cut   = -math.sqrt(2.0 * math.log(math.log(n)))
    shift = {k: fbar[k] if t[k] >= cut else 0.0 for k in live}
    above = 0

    for rows in drawn:
        top = 0.0
        for k in live:
            star  = [f[k][i] for i in rows]
            again = (math.sqrt(kernel_variance(star, q))
                     if any(v != star[0] for v in star) else omega[k])
            top   = max(top, math.sqrt(n) * (mean(star) - shift[k]) / again)
        above += top > big_t

    return white, above / len(drawn)
