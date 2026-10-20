"""The API page's Search section runs as written after its Quickstart, in one namespace,
as a reader copies them: the search holds out the tail of the Quickstart's bars, the one
look at it is counted, and its summary, its consistency and its decay read it beside a
random null of its own space.
"""

from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


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
    assert "search: crossing, frictionless, BTCUSDT 1h, 40 random trials" in said
    assert "  null              mean sharpe " in said and ", look 1 of 1" in said
    assert " of 4 stretches scored above 0: " in said and "decay " in said
