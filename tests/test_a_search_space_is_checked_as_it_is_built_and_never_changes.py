"""A search's space names each parameter's range as an Int, a Float or a Categorical,
each checked as it is built, a wrong type and a wrong value refused with the fix, and
never changed after. Its lattice runs from low to high, a stepped Float's last point high
exactly; its choices reach a trial as plain Python values; and it compares, hashes and
pickles by what it holds.
"""

import math
import pickle

import numpy as np
import pytest

from probatum import (Categorical, Float, Int, ProbatumAttributeError, ProbatumTypeError,
                      ProbatumValueError)

REFUSED = [
    (lambda: Int(1.5, 9), ProbatumTypeError, "low: 1.5; give a whole number, as an int"),
    (lambda: Int(True, 5), ProbatumTypeError, "low: True; give a whole number, as an int"),
    (lambda: Int(1, 9, step="2"), ProbatumTypeError,
     "step: '2'; give a whole number, as an int"),
    (lambda: Int(5, 1), ProbatumValueError, "high: 1, below low, 5; give a high from 5"),
    (lambda: Int(2**63, 2**63 + 10), ProbatumValueError,
     "low: 9223372036854775808, beyond the 64-bit integers a trials table holds; give "
     "one from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807"),
    (lambda: Categorical([2**64, 1]), ProbatumValueError,
     "choices: 18446744073709551616, beyond the 64-bit integers a trials table holds; "
     "give one from -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807"),
    (lambda: Int(1, 10, step=0), ProbatumValueError,
     "step: 0; give a whole number from 1"),
    (lambda: Int(1, 10, step=4), ProbatumValueError,
     "step: 4, which does not divide the span from 1 to 10, 9; give a step that does"),
    (lambda: Int(0, 10, log=True), ProbatumValueError,
     "low: 0 on a log scale; give a low from 1"),
    (lambda: Int(1, 9, step=2, log=True), ProbatumValueError,
     "step: 2 on a log scale; leave it at 1"),
    (lambda: Int(1, 2, log=1), ProbatumTypeError, "log: 1; give True or False"),
    (lambda: Float("a", 1.0), ProbatumTypeError,
     "low: 'a'; give a number, such as 0.5"),
    (lambda: Float(0.0, math.inf), ProbatumValueError, "high: inf; give a finite number"),
    (lambda: Float(1.0, 1.0), ProbatumValueError,
     "high: 1.0, not above low, 1.0; give a high above low"),
    (lambda: Float(0.0, 1.0, step=0), ProbatumValueError,
     "step: 0.0; give a step above 0"),
    (lambda: Float(0.0, 1.0, step=0.3), ProbatumValueError,
     "step: 0.3, which does not divide the span from 0.0 to 1.0; give a step that does"),
    (lambda: Float(0.0, 1e30, step=1e-10), ProbatumValueError,
     "step: 1e-10, which divides the span from 0.0 to 1e+30 into more points than can "
     "be counted; give a coarser step"),
    (lambda: Float(0.0, 1.0, log=True), ProbatumValueError,
     "low: 0.0 on a log scale; give a low above 0"),
    (lambda: Float(0.1, 1.0, step=0.1, log=True), ProbatumValueError,
     "step: 0.1 on a log scale; leave it out"),
    (lambda: Categorical("ab"), ProbatumTypeError,
     "choices: 'ab'; give a list of them, such as [\"ema\", \"sma\"]"),
    (lambda: Categorical({1, 2}), ProbatumTypeError,
     "choices: {1, 2}; give a list of them, such as [\"ema\", \"sma\"]"),
    (lambda: Categorical([]), ProbatumValueError, "choices: none; give at least one"),
    (lambda: Categorical([1, 2, True]), ProbatumValueError,
     "choices: 1 and True, which are one choice; keep one"),
    (lambda: Categorical([math.nan, 0.5, math.nan]), ProbatumValueError,
     "choices: nan and nan, which are one choice; keep one"),
    (lambda: Categorical([1, [2]]), ProbatumTypeError,
     "choices: [2], a list; give None, bools, numbers or strings"),
]


@pytest.mark.parametrize("build, kind, said", REFUSED)
def test_a_range_refuses_a_wrong_type_and_a_wrong_value_with_the_fix(build, kind, said):
    with pytest.raises(kind) as caught:
        build()

    assert type(caught.value) is kind and str(caught.value) == said


def test_a_lattice_runs_from_low_to_high_a_stepped_float_s_last_point_high_exactly():
    # the Specification's worked example: 1, 4, 7 and 10, and ten points from 0.01 to
    # 0.1, where 0.01 + 9 * 0.01 is a hair below 0.1 in binary
    stepped = Int(1, 10, step=3)

    assert [stepped._at(k) for k in range(stepped._size())] == [1, 4, 7, 10]

    ten    = Float(0.01, 0.1, step=0.01)
    points = [ten._at(k) for k in range(ten._size())]

    assert len(points) == 10 and points[-1] == 0.1 and 0.01 + 9 * 0.01 != 0.1
    assert points[:-1] == [0.01 + k * 0.01 for k in range(9)]
    assert Int(3, 3)._size() == 1 and Float(0.0, 1.0)._size() is None
    assert Categorical(["a", "b"])._size() == 2


def test_a_range_holds_plain_python_values():
    numbered = Int(np.int64(1), np.int32(5), step=np.int8(2))

    assert [type(value) for value in numbered._args()] == [int, int, int, bool]

    floated = Float(np.float32(0.5), 1)

    assert (type(floated.low), type(floated.high)) == (float, float)

    choices = Categorical([np.int64(3), np.float32(0.25), np.bool_(False), np.str_("a"),
                           None]).choices

    assert choices == (3, 0.25, False, "a", None)
    assert [type(choice) for choice in choices] == [int, float, bool, str, type(None)]


def test_a_range_never_changes():
    space = Int(1, 5)

    for change in (lambda: setattr(space, "low", 2), lambda: delattr(space, "low")):
        with pytest.raises(ProbatumAttributeError) as caught:
            change()
        assert str(caught.value) == "Int.low: a range does not change; build another"

    assert space.low == 1


def test_a_range_compares_hashes_and_pickles_by_what_it_holds():
    ranges = [Int(1, 10, step=3), Int(1, 1000, log=True), Float(0.01, 0.1, step=0.01),
              Float(0.001, 1.0, log=True), Float(-1.0, 1.0),
              Categorical(["ema", "sma"]), Categorical([1, 2.5, None, math.nan])]

    for given in ranges:
        again = pickle.loads(pickle.dumps(given))
        assert again == given and hash(again) == hash(given) and again is not given

    assert [repr(given) for given in ranges[:6]] == [
        "Int(1, 10, step=3)", "Int(1, 1000, log=True)", "Float(0.01, 0.1, step=0.01)",
        "Float(0.001, 1.0, log=True)", "Float(-1.0, 1.0)", "Categorical(['ema', 'sma'])"]
    assert Int(1, 10) != Int(1, 10, step=3) and Int(1, 10) != Float(1.0, 10.0)
    assert Categorical([1, 2]) != Categorical([1.0, 2.0])
    assert len({Int(1, 10), Int(1, 10), Categorical([1, 2])}) == 2
