"""The API page's Search section runs as written after its Quickstart, in one namespace,
as a reader copies them: the search holds out the tail of the Quickstart's bars, the one
look at it is counted, and its summary, its consistency and its decay read it beside a
random null of its own space.
"""

import re

from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


def printed(said, name, value):
    """Whether ``said`` prints a frame's row of ``name`` holding ``value``."""
    return re.search(rf"^{re.escape(name)} +{re.escape(value)}$", said, re.M) is not None


def test_the_search_section_runs_after_the_quickstart(capsys):
    blocks = sections("Quickstart", "Search")

    assert len(blocks) == 4

    names = {}
    for block in blocks:
        exec(compile(block, str(PAGE), "exec"), names)

    study = names["study"]

    assert study.looks == 1 and names["held"] is study.holdout

    said = capsys.readouterr().out

    assert "Study(40 trials of crossing by random, best sharpe " in said
    assert "Int(40, 200, step=10) Float(0.01, 0.05, step=0.01)" in said
    assert all(printed(said, name, value) for name, value in [
        ("policy", "crossing"), ("venue", "frictionless"), ("data", "BTCUSDT 1h"),
        ("objective", "sharpe"), ("direction", "maximize"), ("sampler", "random"),
        ("trials", "40"), ("looks", "1"), ("unit", "stretches"), ("break_even", "0.0")])
    assert "null_mean" in said and "\nwindow" in said
