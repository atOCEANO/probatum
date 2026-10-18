"""Every error probatum raises is one of its own classes and the built-in it means: a
refusal is caught by its class, by ProbatumError and by that built-in, says its message
as written, a KeyError's unquoted, and crosses a pickle as its class and message. The
kinds sit under the root's class for their built-in, so a wrong value of any topic is a
ProbatumValueError and a policy of the wrong kind a ProbatumTypeError.
"""

import pickle
import re
import sys

import numpy as np
import pytest

import probatum
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (VENUE, one,
                                                                        pair, weights)
from test_smoke import series


def universe_close():
    pair().close


def unknown_instrument():
    pair()["XYZUSDT"]


def bad_bar():
    bars = np.array([[100.0, 99.0, 98.0, 100.0, 1e3]] * 3)
    probatum.Data(bars, time=np.arange(3) * 3_600_000_000_000, interval="1h")


def no_pandas(monkeypatch):
    fills = probatum.backtest(weights(width=1), one(), VENUE).fills
    monkeypatch.setitem(sys.modules, "pandas", None)
    fills.to_pandas()


CASES = [
    (probatum.ProbatumValueError, ValueError,
     lambda monkeypatch: probatum.backtest(weights(), pair(), VENUE, sizing="units"),
     "sizing: 'units'; use \"weight\" or \"quantity\""),
    (probatum.ProbatumTypeError, TypeError,
     lambda monkeypatch: probatum.Venue(name=5),
     "name: 5; give the venue's name as text"),
    (probatum.ProbatumKeyError, KeyError, lambda monkeypatch: unknown_instrument(),
     "data: no instrument 'XYZUSDT'; use one of BTCUSDT, ETHUSDT"),
    (probatum.ProbatumIndexError, IndexError,
     lambda monkeypatch: probatum.backtest(weights(), pair(), VENUE).fills[10**6],
     "row: 1000000 of a table of"),
    (probatum.ProbatumAttributeError, AttributeError,
     lambda monkeypatch: universe_close(),
     "data.close: the universe has 2 instruments; use data.closes"),
    (probatum.ProbatumNotImplementedError, NotImplementedError,
     lambda monkeypatch: probatum.backtest(probatum.Strategy(), pair(), VENUE),
     "next: Strategy does not write it"),
    (probatum.DataError, ValueError, lambda monkeypatch: bad_bar(),
     "is not a bar: open 100, high 99"),
    (probatum.VenueError, ValueError, lambda monkeypatch: probatum.Venue(name=""),
     "name: ''; give the venue a name, or leave it out"),
    (probatum.PolicyError, TypeError,
     lambda monkeypatch: probatum.backtest(probatum.Strategy, pair(), VENUE),
     "policy: got the class Strategy, not an instance; pass Strategy()"),
    (probatum.MissingDependency, ImportError, no_pandas,
     "to_pandas: pandas is not installed; pip install pandas"),
]


@pytest.mark.parametrize("cls, builtin, call, says", CASES,
                         ids=[case[0].__name__ for case in CASES])
def test_a_refusal_is_its_class_probatum_s_and_its_built_in(cls, builtin, call, says,
                                                            monkeypatch):
    with pytest.raises(cls) as caught:
        call(monkeypatch)

    error = caught.value
    assert type(error) is cls
    assert isinstance(error, probatum.ProbatumError) and isinstance(error, builtin)
    assert says in str(error) and not str(error).startswith(("'", '"'))

    again = pickle.loads(pickle.dumps(error))

    assert type(again) is cls and again.args == error.args


def binding_engine(**settings):
    return probatum._probatum.Engine(series(), **settings)


def a_call_with_no_size():
    probatum._probatum.run(series()[:, None, :], {}, {0: [("buy", {})]})


def a_bar_whose_high_is_below_its_open():
    bars = series()
    bars[1, 1] = bars[1, 0] - 1.0
    probatum._probatum.Engine(bars)


def statistics_of_no_year():
    probatum._probatum.stats_of(1.0, np.ones(3), np.ones(1), 0.0, 0.0, 0, 0, 0.0)


# the binding sorts a wrong value by the area it is about: a call's, the run's seed and
# statistics and a batch's are the root's class, a setting's a VenueError and a bar's
# a DataError, and since a kind is also its root each is asserted exactly
BINDING = [
    (probatum.ProbatumValueError,
     lambda: binding_engine(intrabar="random", seed=-1), "seed: -1"),
    (probatum.ProbatumValueError,
     lambda: binding_engine().bracket(take_profit=float("nan")), "take_profit: NaN"),
    (probatum.ProbatumValueError, a_call_with_no_size, "buy has no size"),
    (probatum.ProbatumValueError, statistics_of_no_year, "periods_per_year: 0"),
    (probatum.ProbatumValueError,
     lambda: probatum.Batch(series(), 2).target(np.zeros((3, 1))), "weights: shape"),
    (probatum.VenueError, lambda: binding_engine(funding_rate=float("nan")),
     "funding_rate: NaN"),
    (probatum.DataError, a_bar_whose_high_is_below_its_open, "is not a bar"),
]


@pytest.mark.parametrize("cls, call, says", BINDING, ids=[
    "seed", "verb", "script", "statistics", "batch", "setting", "bar"])
def test_the_binding_sorts_a_wrong_value_by_what_it_is_about(cls, call, says):
    with pytest.raises(cls, match=re.escape(says)) as caught:
        call()

    assert caught.type is cls


def test_each_kind_sits_under_the_root_s_class_for_its_built_in():
    for kind in (probatum.DataError, probatum.VenueError, probatum.SpecError,
                 probatum.LookaheadError):
        assert issubclass(kind, probatum.ProbatumValueError)

    assert issubclass(probatum.PolicyError, probatum.ProbatumTypeError)
    assert issubclass(probatum.MissingDependency, ImportError)
    assert issubclass(probatum.ProbatumWarning, UserWarning)
