"""A validation's training piece reads no bar of a test span: a piece after a span is a
Data of its own that holds its rows alone, and its account opens past the embargo, the
fill window and its warm-up while its init reads every row of it. A fit of several
pieces is scored on their returns joined end to end, as the Specification works it, a
fit of one piece is a search's trial on it, and a fit's trials run in the workers as
they run here.
"""

import numpy as np
import pytest

import probatum
from probatum import _workers
from probatum._backtest import _annualised, _markets
from probatum._data import _before, _piece
from probatum._probatum import Engine
from probatum._search import _joined, _pieces, _Runner
from probatum._strategy import _play, _Run
from probatum._venue import _run
from test_a_data_from_the_router_is_every_bar_it_asked_for import (Router, bars, fetch,
                                                                   settlements)
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import VENUE, one, pair
from test_a_strategy_sees_its_bar_and_its_account_and_nothing_later import Cross


class Reader(probatum.Strategy):
    """The first and last time its init was handed, every row it decided at and every
    time its history read."""


    def __init__(self, warmup=0):
        self.warmup = warmup


    def init(self, data):
        self.first, self.last = data.time[0], data.time[-1]
        self.rows, self.read = [], set()


    def next(self, bar, account):
        self.rows.append(bar.i)
        self.read.update(bar.history().time.tolist())


class Ran:
    """A run's records, as a fit's join reads them."""


    def __init__(self, curve, pnl, held, fills, funding=0.0):
        self.curve, self.pnl, self.held, self.fills = curve, pnl, held, fills
        self.funding = funding


    def equity_curve(self):
        return np.array(self.curve)


    def trades_since(self, first):
        return [{"pnl": pnl} for pnl in self.pnl]


    def stats(self, rate, risk_free):
        return {"num_fills": self.fills, "funding_paid": self.funding}


    def held_steps(self):
        return self.held


def test_a_piece_after_a_span_holds_its_rows_and_no_other_bar():
    data  = pair()
    piece = _piece(data, 21, 35)

    assert len(piece) == 14 and piece.names == data.names
    assert len(piece._root.time) == 14
    np.testing.assert_array_equal(piece.time, data.time[21:35])
    np.testing.assert_array_equal(piece.closes, data.closes[21:35])
    assert piece._root.funded == _before(data, 35)._root.funded


def test_a_piece_holds_the_funding_its_rows_own_and_no_earlier_event():
    held  = {"BTCUSDT": bars(48)}
    data  = fetch(Router(held, {"BTCUSDT": settlements()}), held, funding=True)
    piece = _piece(data, 21, 35)

    np.testing.assert_array_equal(piece.funding, data[21:35].funding)
    assert 0 < len(piece.funding) < len(_before(data, 35).funding)
    assert piece._root.funded == frozenset({0})


def test_a_piece_opens_past_its_embargo_the_fill_window_and_its_warm_up():
    piece  = _piece(pair(), 5, 40)
    run    = _run(VENUE, piece)
    engine = Engine(run.candles, report=True, **run.settings)

    for later, warmup in ((10, 3), (4, 12)):
        played       = _Run(engine, piece, run.start, _markets(run))
        played.later = later
        reader       = Reader(warmup)
        _play(reader, played)

        opens = max(later, warmup)
        assert (reader.first, reader.last) == (piece.time[0], piece.time[-1])
        assert reader.rows[0] == opens and min(reader.read) == piece.time[0]
        assert len(engine.equity_curve()) == len(piece) - 1 - opens


def test_a_trial_whose_warm_up_leaves_no_piece_two_rows_fails():
    data   = pair()
    runner = _Runner(Reader, "class", data, _run(VENUE, data), _annualised(data), 0.0,
                     "sharpe", 0, pieces=_pieces(data, VENUE, [(30, 30, 40, 5)]))

    trial = runner.trial(0, {"warmup": 9})

    assert trial.state == "failed"
    assert trial.error.startswith("ProbatumValueError: policy: no piece of the fit "
                                  "holds two rows past its warm-up with {'warmup': 9}")


def test_two_pieces_join_end_to_end_as_the_page_works_it():
    first  = Ran([10_500.0, 11_000.0], [1_000.0], held=2, fills=2, funding=5.0)
    second = Ran([9_950.0, 9_900.0], [-100.0], held=2, fills=2, funding=2.0)
    stats  = _joined([first, second], 10_000.0, 8_760.0, 0.0)

    assert stats["total_return_pct"] == pytest.approx(8.9)
    assert (stats["num_trades"], stats["win_rate"], stats["num_fills"]) == (2, 0.5, 4)
    # the round trips as each piece made them, unscaled, and the funding of each
    assert stats["profit_factor"] == pytest.approx(10.0)
    assert stats["funding_paid"]  == pytest.approx(7.0)
    assert stats["exposure_pct"]  == 100.0

    # a join stops at the first piece that ends at or below zero equity, the later
    # pieces adding their steps at that equity and nothing else
    bust  = Ran([5_000.0, 0.0], [-10_000.0], held=2, fills=2, funding=3.0)
    stats = _joined([bust, second], 10_000.0, 8_760.0, 0.0)

    assert stats["total_return_pct"] == -100.0
    assert (stats["num_trades"], stats["num_fills"], stats["exposure_pct"]) == (1, 2,
                                                                                50.0)
    assert stats["funding_paid"] == pytest.approx(3.0)


def test_a_join_counts_each_run_s_steps_in_a_position_as_its_engine_does():
    data   = one()
    rate   = _annualised(data)
    runner = _Runner(Cross, "class", data, _run(VENUE, data), rate, 0.0, "sharpe", 0,
                     pieces=_pieces(data, VENUE, [(0, 0, 20, None), (22, 22, 40, 3)]))

    assert runner.trial(0, {"fast": 3, "slow": 8}).state == "complete"

    engines = [runner.engines[0], runner.engines[1]]
    steps   = [len(engine.equity_curve()) for engine in engines]

    for engine, counted in zip(engines, steps):
        exposure = engine.stats(rate, 0.0)["exposure_pct"]
        assert 0 < engine.held_steps() <= counted
        assert exposure == pytest.approx(100.0 * engine.held_steps() / counted)

    held   = sum(engine.held_steps() for engine in engines)
    joined = _joined(engines, runner.opening, rate, 0.0)

    assert joined["exposure_pct"] == pytest.approx(100.0 * held / sum(steps))


def test_a_fit_of_one_piece_is_a_search_s_trial_on_it():
    data   = one()
    rate   = _annualised(data)
    piece  = _before(data, 30)[5:]
    alone  = _Runner(Cross, "class", piece, _run(VENUE, piece), rate, 0.0, "sharpe", 0)
    fitted = _Runner(Cross, "class", data, _run(VENUE, data), rate, 0.0, "sharpe", 0,
                     pieces=_pieces(data, VENUE, [(0, 5, 30, None)]))
    params = {"fast": 3, "slow": 8}
    trial  = fitted.trial(0, params)

    assert trial.state == "complete" and trial.stats == alone.trial(0, params).stats


def test_a_fit_s_trials_run_in_the_workers_as_they_run_here(tmp_path):
    pytest.importorskip("cloudpickle")

    data     = one()
    rate     = _annualised(data)
    specs    = [(0, 0, 15, None), (22, 22, 40, 3)]
    settings = dict(kind="class", made="strategy", objective="sharpe", min_trades=0,
                    rate=rate, risk_free=0.0, keep="none", fits=[specs])

    _workers._wrote(tmp_path, Cross, VENUE, data, settings)
    pool = _workers._Pool(tmp_path, 2, 2)
    try:
        pool.fit = 0
        far      = pool.run([(0, {"fast": 3, "slow": 8}), (1, {"fast": 4, "slow": 10})])
    finally:
        pool.close()

    here = _Runner(Cross, "class", data, _run(VENUE, data), rate, 0.0, "sharpe", 0,
                   pieces=_pieces(data, VENUE, specs))

    assert [trial.state for trial in far] == ["complete", "complete"]
    for trial in far:
        assert trial.stats == here.trial(trial.number, trial.params).stats
