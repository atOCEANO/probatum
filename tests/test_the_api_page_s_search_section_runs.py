"""The API page's Search section runs as written after its Quickstart, in one namespace,
as a reader copies them: the search holds out the tail of the Quickstart's bars, and the
one look at it is counted.
"""

from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


def test_the_search_section_runs_after_the_quickstart(capsys):
    blocks = sections("Quickstart", "Search")

    assert len(blocks) == 3

    names = {}
    for block in blocks:
        exec(compile(block, str(PAGE), "exec"), names)

    study = names["study"]

    assert study.looks == 1 and names["held"] is study.holdout

    said = capsys.readouterr().out

    assert "Study(40 trials of crossing by random, best sharpe " in said
    assert "Int(40, 200, step=10) Float(0.01, 0.05, step=0.01)" in said
