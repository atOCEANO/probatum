"""A splitter divides bars and trains on no row whose label reaches a test span: each
split's rows are exactly those the Specification's rule keeps, read off it one row at
a time over random ends and universes, and its worked examples hold. PurgedKFold is
scikit-learn's KFold where no label crosses a group, CombinatorialPurged of one test
group is PurgedKFold, its paths take each group from the split the rule names, and
scikit-learn's own tools take every splitter as it is.
"""

import copy
import itertools
import math
import pickle
import re

import numpy as np
import pytest

import probatum
from probatum import (ProbatumAttributeError, ProbatumTypeError, ProbatumValueError,
                      ProbatumWarning)
from probatum.cv import CombinatorialPurged, Holdout, PurgedKFold, WalkForward

NONE = np.iinfo(np.int64).min


def groups(bars, n):
    """Group edges as KFold sizes folds, written out again."""
    sizes = [bars // n + (1 if g < bars % n else 0) for g in range(n)]
    return [sum(sizes[:g]) for g in range(n + 1)]


def layout(splitter, bars):
    """Each split's test spans and training ranges, from the Specification's text."""
    if isinstance(splitter, Holdout):
        held = (splitter.tail if isinstance(splitter.tail, int)
                else round(bars * splitter.tail))
        return [([(bars - held, bars - 1)], [(0, bars - held)])]

    if isinstance(splitter, WalkForward):
        count = (bars - splitter.train) // splitter.test
        out   = []
        for j in range(count):
            first = splitter.train + j * splitter.test
            last  = bars - 1 if j == count - 1 else first + splitter.test - 1
            out.append(([(first, last)],
                        [(0 if splitter.anchored else first - splitter.train, first)]))
        return out

    n, tested = ((splitter.k, [(g,) for g in range(splitter.k)])
                 if isinstance(splitter, PurgedKFold)
                 else (splitter.groups, list(itertools.combinations(
                     range(splitter.groups), splitter.test_groups))))
    edges = groups(bars, n)
    out   = []
    for chosen in tested:
        spans = []
        for g in chosen:
            if spans and spans[-1][1] == edges[g] - 1:
                spans[-1] = (spans[-1][0], edges[g + 1] - 1)
            else:
                spans.append((edges[g], edges[g + 1] - 1))
        trains = [(edges[g], edges[g + 1]) for g in range(n) if g not in chosen]
        out.append((spans, trains))

    return out


def by_rule(splitter, end, bars):
    """Each split's training and test rows, one row at a time as the rule reads: a row
    tests where its bar is in a span and its label ends by the last bar, and trains
    where its bar is a training bar and its label ends by the last bar,
    unless its label reaches a span from before it or it starts after a span up to the
    span's reach and the embargo."""
    width    = end.shape[1]
    labelled = [(t, k) for t in range(bars) for k in range(width)
                if end[t, k] != NONE and end[t, k] < bars]

    embargo = getattr(splitter, "embargo", 0)
    if embargo is None:
        embargo = max((end[t, k] - t - 1 for t, k in labelled), default=0)

    out = []
    for spans, trains in layout(splitter, bars):
        reaches = []
        for a, b in spans:
            ends = [end[t, k] for t in range(a, b + 1) for k in range(width)
                    if end[t, k] != NONE]
            reaches.append((a, b, max([b] + ends)))

        train, test = [], []
        for t, k in labelled:
            row = t * width + k
            if any(a <= t <= b for a, b, _ in reaches):
                test.append(row)
            elif (any(lo <= t < hi for lo, hi in trains)
                  and not any(t < a <= end[t, k] for a, _, _ in reaches)
                  and not any(b < t <= reach + embargo for _, b, reach in reaches)):
                train.append(row)
        out.append((train, test))

    return out


def random_end(rng, bars, width):
    """Label ends at a few bars after each row, now and then far after, past the last
    bar or none at all."""
    t   = np.arange(bars)[:, None]
    end = t + 1 + rng.integers(0, rng.integers(1, 12), size=(bars, width))
    far = rng.random((bars, width)) < 0.05
    end[far] += rng.integers(10, 60, size=far.sum())
    end[rng.random((bars, width)) < 0.05] = NONE
    return end.astype(np.int64)


def random_splitter(rng, bars, end):
    embargo = [None, 0, int(rng.integers(1, 8))][rng.integers(3)]
    kind    = rng.integers(4)

    if kind == 0:
        return PurgedKFold(int(rng.integers(2, 7)), embargo=embargo, end=end)

    if kind == 1:
        n = int(rng.integers(3, 7))
        return CombinatorialPurged(n, int(rng.integers(1, n)), embargo=embargo, end=end)

    if kind == 2:
        train = int(rng.integers(5, bars // 2))
        return WalkForward(
            train,
            int(rng.integers(3, bars // 4)),
            anchored = bool(rng.integers(2)),
            end      = end,
        )

    tail = float(rng.uniform(0.1, 0.5)) if rng.integers(2) else int(rng.integers(2, 20))
    return Holdout(tail, end=end)


@pytest.mark.parametrize("seed", range(6))
def test_each_split_trains_and_tests_on_exactly_the_rows_the_rule_keeps(seed):
    rng     = np.random.default_rng(seed)
    checked = 0

    while checked < 40:
        bars, width = int(rng.integers(30, 120)), int(rng.integers(1, 4))
        end = random_end(rng, bars, width)
        try:
            splitter = random_splitter(rng, bars, end if width > 1 or rng.integers(2)
                                       else end[:, 0])
        except ProbatumValueError:
            continue

        wanted = by_rule(splitter, end, bars)
        if any(not train or not test for train, test in wanted):
            with pytest.raises(ProbatumValueError, match="no row"):
                splitter.split(np.zeros(bars * width))
            continue

        got = list(splitter.split(np.zeros(bars * width)))

        assert splitter.get_n_splits(np.zeros(bars * width)) == len(got) == len(wanted)
        for (train, test), (want_train, want_test) in zip(got, wanted):
            assert train.dtype == test.dtype == np.int64
            assert train.tolist() == want_train and test.tolist() == want_test

        checked += 1


def test_the_purge_s_worked_examples_train_on_the_bars_the_page_says():
    t = np.arange(1_000)
    train, test = list(PurgedKFold(5, embargo=24, end=t + 25).split(t))[2]

    assert (test.min(), test.max()) == (400, 599)
    assert train.tolist() == list(range(0, 375)) + list(range(649, 975))

    end = t + 11
    end[590], end[599] = 640, 610
    train, _ = list(PurgedKFold(5, embargo=0, end=end).split(t))[2]

    assert train.tolist() == list(range(0, 389)) + list(range(641, 989))


def test_the_layouts_worked_examples_hold():
    edges = PurgedKFold(5)._paths(1_003)[0]
    assert [(a, b) for a, b, _ in edges] == [(0, 200), (201, 401), (402, 602),
                                             (603, 802), (803, 1_002)]

    rolling, anchored = WalkForward(400, 150), WalkForward(400, 150, anchored=True)
    assert [tests for tests, _ in rolling._layout(1_000)] == [
        [(400, 549)], [(550, 699)], [(700, 849)], [(850, 999)]]
    assert rolling._layout(1_000)[0][1]   == [(0, 400)]
    assert rolling._layout(1_000)[-1][1]  == [(450, 850)]
    assert anchored._layout(1_000)[-1][1] == [(0, 850)]
    assert rolling._layout(1_030)[-1][0]  == [(850, 1_029)]

    paths = CombinatorialPurged(6, 2)._paths(600)
    assert [[split for _, _, split in path] for path in paths] == [
        [0, 0, 1, 2, 3, 4], [1, 5, 5, 6, 7, 8], [2, 6, 9, 9, 10, 11],
        [3, 7, 10, 12, 12, 13], [4, 8, 11, 13, 14, 14]]

    feeds = {split: {p for p, path in enumerate(paths) for *_, s in path if s == split}
             for split in range(15)}
    assert {split for split, fed in feeds.items() if len(fed) == 1} == {0, 5, 9, 12, 14}
    assert all(sum(s == split for path in paths for *_, s in path) == 2
               for split in range(15))


def test_one_test_group_combines_into_purged_k_fold_s_splits():
    rng = np.random.default_rng(3)
    end = (np.arange(90)[:, None] + 1 + rng.integers(0, 5, (90, 2))).astype(np.int64)
    end[rng.random((90, 2)) < 0.05] = NONE
    rows = np.zeros(180)

    for k in (2, 3, 5):
        one = [(train.tolist(), test.tolist()) for train, test in
               CombinatorialPurged(k, 1, embargo=3, end=end).split(rows)]
        fold = [(train.tolist(), test.tolist()) for train, test in
                PurgedKFold(k, embargo=3, end=end).split(rows)]
        assert one == fold
        assert len(CombinatorialPurged(k, 1)._paths(90)) == 1


def test_purged_k_fold_with_no_label_is_scikit_learn_s_k_fold():
    model_selection = pytest.importorskip("sklearn.model_selection")
    rows = np.zeros(103)

    for k in (2, 4, 7):
        ours = [(train.tolist(), test.tolist())
                for train, test in PurgedKFold(k).split(rows)]
        theirs = [(train.tolist(), test.tolist())
                  for train, test in model_selection.KFold(k).split(rows)]
        assert ours == theirs


def test_scikit_learn_s_tools_take_every_splitter_as_it_is():
    model_selection = pytest.importorskip("sklearn.model_selection")
    linear_model    = pytest.importorskip("sklearn.linear_model")
    rng             = np.random.default_rng(5)

    bars, width = 240, 2
    end = (np.arange(bars)[:, None] + 1 + rng.integers(0, 5, (bars, width))).astype(
        np.int64)
    end[-6:] = NONE

    x          = rng.normal(size=(bars * width, 3))
    y          = x @ np.array([0.5, -0.2, 0.1]) + rng.normal(scale=0.1, size=bars * width)
    unlabelled = set(np.flatnonzero(end.reshape(-1) == NONE).tolist())
    splitters  = [Holdout(0.25, end=end), WalkForward(80, 40, end=end),
                  PurgedKFold(4, embargo=2, end=end),
                  CombinatorialPurged(5, 2, embargo=2, end=end)]

    for splitter in splitters:
        assert not any(unlabelled & set(rows.tolist())
                       for split in splitter.split(x) for rows in split)
        scores = model_selection.cross_validate(linear_model.Ridge(), x, y, cv=splitter)
        assert len(scores["test_score"]) == splitter.get_n_splits(x)
        search = model_selection.GridSearchCV(linear_model.Ridge(),
                                              {"alpha": [0.1, 1.0]}, cv=splitter)
        assert search.fit(x, y).n_splits_ == splitter.get_n_splits(x)
        for again in (copy.deepcopy(splitter), pickle.loads(pickle.dumps(splitter))):
            assert again == splitter and hash(again) == hash(splitter)
            assert [test.tolist() for _, test in again.split(x)] == [
                test.tolist() for _, test in splitter.split(x)]

    for splitter in (PurgedKFold(4), CombinatorialPurged(4, 1)):
        rows      = rng.normal(size=(100, 3))
        predicted = model_selection.cross_val_predict(
            linear_model.Ridge(), rows, rows @ np.ones(3), cv=splitter)
        assert predicted.shape == (100,)

    # built with end, the last bar's rows, whose labels end past it, are in no test
    # set, though every row has a label
    rows = rng.normal(size=(240, 3))
    with pytest.raises(ValueError, match="partitions"):
        model_selection.cross_val_predict(linear_model.Ridge(), rows, rows @ np.ones(3),
                                          cv=PurgedKFold(4, end=np.arange(240) + 3))

    sklearn = pytest.importorskip("sklearn")
    with sklearn.config_context(enable_metadata_routing=True):
        scores = model_selection.cross_validate(linear_model.Ridge(), x, y,
                                                cv=PurgedKFold(4, end=end))
        assert len(scores["test_score"]) == 4


def test_a_split_the_purge_leaves_with_nothing_is_refused_before_any_is_given():
    t = np.arange(100)

    with pytest.raises(ProbatumValueError, match=re.escape(
            "split 0, testing group 0: no row left to train on once the purge and an "
            "embargo of 60 bars took the rest; give more bars, a smaller k or a "
            "shorter embargo")):
        PurgedKFold(2, embargo=60, end=t + 1).split(t)

    end = t + 1
    end[80:] = NONE

    with pytest.raises(ProbatumValueError, match=re.escape(
            "window 0, testing bars 80 to 99: no row to test on, as none of its rows "
            "holds a label that ends by the last bar; give them labels that do")):
        WalkForward(80, 20, end=end).split(t)


@pytest.mark.parametrize("build, error, says", [
    (lambda: PurgedKFold(1), ProbatumValueError, "k: 1; give a whole number from 2"),
    (lambda: PurgedKFold(2.5), ProbatumTypeError, "k: 2.5; give a whole number from 2"),
    (lambda: PurgedKFold(True), ProbatumTypeError, "k: True; give a whole number"),
    (lambda: PurgedKFold(3, embargo=-1), ProbatumValueError,
     "embargo: -1; give a whole number of bars from 0"),
    (lambda: PurgedKFold(3, embargo=1.5), ProbatumTypeError,
     "embargo: 1.5; give a whole number of bars from 0"),
    (lambda: CombinatorialPurged(4, 4), ProbatumValueError,
     "test_groups: 4; give a whole number from 1 to 3"),
    (lambda: CombinatorialPurged(1, 1), ProbatumValueError,
     "groups: 1; give a whole number from 2"),
    (lambda: WalkForward(0, 10), ProbatumValueError,
     "train: 0; give a whole number of bars from 1"),
    (lambda: WalkForward(10, 0.5), ProbatumTypeError,
     "test: 0.5; give a whole number of bars from 1"),
    (lambda: WalkForward(10, 5, anchored="yes"), ProbatumTypeError,
     "anchored: 'yes'; give True or False"),
    (lambda: Holdout(1.5), ProbatumValueError,
     "tail: 1.5; give a share of the bars above 0 and below 1"),
    (lambda: Holdout("0.2"), ProbatumTypeError,
     "tail: '0.2'; give a share of the bars"),
    (lambda: Holdout(0), ProbatumValueError,
     "tail: 0; give a whole number of bars from 1"),
    (lambda: PurgedKFold(3, end=np.arange(10.0) + 1), ProbatumTypeError,
     "end: an array of float64; give each row's label end as an int64 bar"),
    (lambda: PurgedKFold(3, end=np.array([1, 2, 2**64 - 1], dtype=np.uint64)),
     ProbatumValueError,
     "end: 18446744073709551615, past the largest int64; give each row's label end as "
     "an int64 bar"),
    (lambda: PurgedKFold(3, end=np.ones((2, 2, 2), dtype=np.int64)), ProbatumValueError,
     "end: an array of shape (2, 2, 2); give (T,) for one instrument or (T, N)"),
    (lambda: PurgedKFold(3, end=np.arange(10)), ProbatumValueError,
     "end: row 0 ends at 0, at or before its own bar; give the bar its trade closed "
     "on, from the next on, or the least int64 for a row with no label, and a (T, N) "
     "grid flattened reads its rows as bars, so pass it as (T, N)"),
    (lambda: PurgedKFold(3, end=np.array([[1, 0], [2, 2], [3, 3]])), ProbatumValueError,
     "end: row 0 of instrument 1 ends at 0, at or before its own bar"),
    (lambda: PurgedKFold(5, end=np.arange(1, 4)), ProbatumValueError,
     "k: 5 for 3 bars, which leaves a group with none; give k up to 3"),
    (lambda: WalkForward(90, 20, end=np.arange(1, 101)), ProbatumValueError,
     "train: 90 and test: 20 for 100 bars, which leave no window; give fewer bars to "
     "train on or to test"),
    (lambda: Holdout(0.001, end=np.arange(1, 401)), ProbatumValueError,
     "tail: 0.001 of 400 bars rounds to 0; give a larger share or a number of bars "
     "from 1"),
    (lambda: Holdout(400, end=np.arange(1, 401)), ProbatumValueError,
     "tail: 400 of 400 bars, which leaves none to train on; give fewer"),
])
def test_a_splitter_it_cannot_build_is_refused_naming_the_fix(build, error, says):
    with pytest.raises(error, match=re.escape(says)) as caught:
        build()

    assert caught.type is error


def test_rows_it_was_not_built_for_and_groups_it_ignores_are_named():
    splitter = PurgedKFold(3, end=np.repeat(np.arange(1, 26)[:, None], 2, axis=1))

    with pytest.raises(ProbatumValueError, match=re.escape(
            "X: 40 rows where this splitter was built for 50; build it for the rows it "
            "splits")):
        splitter.split(np.zeros(40))
    with pytest.raises(ProbatumValueError, match="X: 40 rows"):
        splitter.get_n_splits(np.zeros(40))
    with pytest.raises(ProbatumTypeError, match="X: int, which has no rows"):
        splitter.split(5)
    with pytest.warns(ProbatumWarning, match=re.escape(
            "groups passed to split: ignored by PurgedKFold, which divides by time")):
        list(splitter.split(np.zeros(50), groups=np.zeros(50)))
    with pytest.raises(ProbatumValueError, match=re.escape(
            "X: none, and a WalkForward built without end counts its windows on the "
            "rows it splits; pass X, or build it with end=")):
        WalkForward(10, 5).get_n_splits()

    assert (Holdout(0.2).get_n_splits(), PurgedKFold(4).get_n_splits(),
            CombinatorialPurged(8, 2).get_n_splits()) == (1, 4, 28)


def test_a_splitter_never_changes_and_prints_as_its_call():
    end      = np.arange(1, 1_001).reshape(500, 2)
    splitter = PurgedKFold(5, embargo=24, end=end)

    end[0, 0] = 400

    assert splitter.end[0, 0] == 1
    with pytest.raises(ProbatumAttributeError, match="PurgedKFold.k: a splitter does "
                                                     "not change; build another"):
        splitter.k = 3
    with pytest.raises(ValueError):
        splitter.end[0, 0] = 9

    assert repr(splitter) == "PurgedKFold(5, embargo=24, end of shape (500, 2))"
    assert repr(CombinatorialPurged(8, 2)) == "CombinatorialPurged(8, 2)"
    assert repr(WalkForward(400, 150, anchored=True)) == (
        "WalkForward(400, 150, anchored=True)")
    assert repr(Holdout(0.2)) == "Holdout(0.2)"
    assert PurgedKFold(5) == PurgedKFold(5) != PurgedKFold(5, embargo=0)
    assert probatum.cv.PurgedKFold is PurgedKFold
    assert not any(hasattr(splitter, name) for name in ("shuffle", "random_state",
                                                       "get_params"))
    assert math.comb(8, 2) == CombinatorialPurged(8, 2).get_n_splits()
