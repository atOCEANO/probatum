"""Every estimator the Inference section takes from a paper gives that paper's numbers.

Each test hands the reference statistics what a paper tabulates and reads back what it
prints: Lo's standard errors and his Sharpe ratios of summed returns, Bailey and Lopez
de Prado's probabilistic Sharpe and minimum track records, the stationary block lengths
of Patton, Politis and White's correction, Memmel's variance of a difference of two
Sharpe ratios, and sandwich's Newey-West covariance on its Investment data.
`probatum.stats` is held to the reference by its own agreement test.
"""

import math

import numpy as np
import pytest

from reference import statistics


@pytest.mark.parametrize("shape, count, printed", [
    (0.5, 12, 0.306), (1.5, 60, 0.188), (3.0, 60, 0.303), (2.0, 500, 0.077)])
def test_lo_s_standard_errors_are_the_lag_zero_variance_at_normal_moments(shape, count,
                                                                          printed):
    # Lo (2002) table 1 divides by T where the page divides by n - 1, so the count is
    # his and only the variance is the page's
    variance = statistics.lag_zero(shape, 0.0, 3.0)

    assert round(math.sqrt(variance / count), 3) == printed


@pytest.mark.parametrize("rho, q, printed", [
    (0.2, 12, 2.88), (-0.2, 12, 4.17), (0.0, 12, 3.46), (0.9, 250, 3.70), (-0.9, 2, 4.47)])
def test_lo_s_sharpe_of_summed_returns_is_the_bartlett_sum_at_lag_q_minus_1(rho, q,
                                                                            printed):
    # Lo (2002) table 2: eta(q) for returns correlated rho^k at lag k
    ratio = statistics.bartlett([rho ** k for k in range(q)], q - 1)

    assert round(math.sqrt(q / ratio), 2) == printed


def test_bailey_and_lopez_de_prado_s_probabilistic_sharpe_is_0_913_over_24_months():
    variance = statistics.lag_zero(0.458, -2.448, 10.164)

    assert round(statistics.probability(0.458, 0.0, variance, 24), 3) == 0.913
    assert round(statistics.probability(0.458, 0.0, statistics.lag_zero(0.458, 0.0, 3.0),
                                        24), 3) == 0.982
    # the inputs are printed rounded, and at 36 months they give 0.95351, on the edge of
    # the printed 0.953: half a unit in the Sharpe's third decimal moves it by 0.0002
    assert statistics.probability(0.458, 0.0, variance, 36) == pytest.approx(0.953,
                                                                              abs=1e-3)


def test_their_minimum_track_record_is_59_895_months_and_reaches_95_percent_there():
    shape, benchmark = 2.0 / math.sqrt(12.0), 1.0 / math.sqrt(12.0)
    variance = statistics.lag_zero(shape, -0.72, 5.78)
    months   = statistics.track_record(shape, benchmark, variance, 0.95)

    assert round(months, 3) == 59.895
    assert statistics.probability(shape, benchmark, variance, months) == pytest.approx(0.95)


@pytest.mark.parametrize("periods, printed", [(252, 2.73), (52, 2.83), (12, 3.24)])
def test_their_track_records_for_2_against_1_at_normal_moments(periods, printed):
    # figures 8 to 10: an annual Sharpe of 2 against 1, daily, weekly and monthly
    shape, benchmark = 2.0 / math.sqrt(periods), 1.0 / math.sqrt(periods)
    variance = statistics.lag_zero(shape, 0.0, 3.0)
    years    = statistics.track_record(shape, benchmark, variance, 0.95) / periods

    assert round(years, 2) == printed


@pytest.mark.parametrize("rho, count, printed, circular", [
    (0.7, 200, 11.47, 13.12), (0.7, 800, 18.20, 20.83), (0.1, 200, 2.01, None),
    (0.1, 800, 3.20, 3.66), (-0.4, 200, 5.66, 6.48), (-0.4, 800, 8.99, None)])
def test_the_correction_s_block_lengths_follow_from_g_over_g_0(rho, count, printed,
                                                              circular):
    # Patton, Politis and White (2009) table 1. Two printed circular lengths, 2.31 and
    # 10.23, do not follow from (3/2)^(1/3) times the stationary one, which gives 2.30
    # and 10.29, and are left out
    length = statistics.stationary(2.0 * rho / (1.0 - rho * rho), count)

    assert round(length, 2) == printed
    if circular is not None:
        assert round(1.5 ** (1.0 / 3.0) * length, 2) == circular


@pytest.mark.parametrize("sa, sb, rho", [
    (0.10, 0.05, 0.5), (0.30, -0.10, 0.9), (0.0, 0.2, -0.4), (0.15, 0.15, 0.0)])
def test_memmel_s_variance_of_a_difference_is_the_lag_zero_variance_of_its_influence(
        sa, sb, rho):
    # the mean of (z_a - z_b)^2 over two normal returns correlated rho, by
    # Gauss-Hermite quadrature, which is exact for a polynomial of degree 4
    nodes, weights = np.polynomial.hermite_e.hermegauss(5)
    weights = weights / math.sqrt(2.0 * math.pi)
    total   = 0.0

    for e1, w1 in zip(nodes, weights):
        for e2, w2 in zip(nodes, weights):
            a  = sa + e1
            b  = sb + rho * e1 + math.sqrt(1.0 - rho * rho) * e2
            za = statistics.influence_at([a], sa, 1.0)[0]
            zb = statistics.influence_at([b], sb, 1.0)[0]
            total += w1 * w2 * (za - zb) ** 2

    memmel = 2.0 - 2.0 * rho + (sa * sa + sb * sb - 2.0 * sa * sb * rho * rho) / 2.0
    assert total == pytest.approx(memmel, rel=1e-12)


# sandwich's Investment data: GNP, investment, the price index and the interest rate
INVESTMENT = [
    (596.7, 90.9, 0.7167, 3.23), (637.7, 97.4, 0.7277, 3.55), (691.1, 113.5, 0.7436, 4.04),
    (756.0, 125.7, 0.7676, 4.5), (799.6, 122.8, 0.7906, 4.19), (873.4, 133.3, 0.8254, 5.16),
    (944.0, 149.3, 0.8679, 5.87), (992.7, 144.2, 0.9145, 5.95), (1077.6, 166.4, 0.9601, 4.88),
    (1185.9, 195.0, 1.0, 4.5), (1326.4, 229.8, 1.0575, 6.44), (1434.2, 228.7, 1.1508, 7.83),
    (1549.2, 206.1, 1.2579, 6.25), (1718.0, 257.9, 1.3234, 5.5), (1918.3, 324.1, 1.4005, 5.46),
    (2163.9, 386.6, 1.5042, 7.46), (2417.8, 423.0, 1.6342, 10.28),
    (2631.7, 401.9, 1.7842, 11.77), (2954.1, 474.9, 1.9514, 13.42),
    (3073.0, 414.5, 2.0688, 11.02)]


def test_sandwich_s_newey_west_covariance_of_the_investment_regression_at_lag_4():
    # NeweyWest(lm(RealInv ~ RealGNP + RealInt), lag = 4, prewhite = FALSE); the first
    # year has no real interest rate, which needs the price a year before
    rows, invested = [], []
    for before, now in zip(INVESTMENT, INVESTMENT[1:]):
        gnp, investment, price, interest = now
        real_interest = interest - 100.0 * (price - before[2]) / before[2]
        rows.append([1.0, gnp / price, real_interest])
        invested.append(investment / price)

    coef, residuals = statistics.least_squares(rows, invested)
    covariance = statistics.newey_west(rows, residuals, 4)

    printed = {
        (0, 0): (359.4170681, 7),
        (1, 1): (0.0002805888, 10),
        (2, 2): (11.171472998, 9),
        (0, 1): (-0.3115505035, 10),
        (0, 2): (-4.089319305, 9),
        (1, 2): (-0.005355931, 9),
    }
    for (i, j), (value, places) in printed.items():
        assert covariance[i][j] == pytest.approx(value, abs=0.5 * 10.0 ** -places)
