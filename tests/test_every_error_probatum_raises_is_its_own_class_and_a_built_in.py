"""Every error probatum raises is one of its own classes and the built-in it means: a
refusal is caught by its class, by ProbatumError and by that built-in, says its message
as written, a KeyError's unquoted, and crosses a pickle as its class and message. The
kinds sit under the root's class for their built-in, so a wrong value of any topic is a
ProbatumValueError and a policy of the wrong kind a ProbatumTypeError.
"""

import pickle
import sys

import numpy as np
import pytest

import probatum
from test_a_policy_is_read_one_way_and_an_array_runs_in_one_call import (VENUE, one,
                                                                        pair, weights)


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


def test_each_kind_sits_under_the_root_s_class_for_its_built_in():
    for kind in (probatum.DataError, probatum.VenueError, probatum.SpecError,
                 probatum.LookaheadError):
        assert issubclass(kind, probatum.ProbatumValueError)

    assert issubclass(probatum.PolicyError, probatum.ProbatumTypeError)
    assert issubclass(probatum.MissingDependency, ImportError)
    assert issubclass(probatum.ProbatumWarning, UserWarning)
