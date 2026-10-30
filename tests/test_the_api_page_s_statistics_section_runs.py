"""The API page's Statistics of a run section runs as written after its Quickstart, in
one namespace, as a reader copies them, and prints the frames the page shows beneath
them: the Quickstart's run's deepest drawdowns, its time under water and the run beside
holding the coin, then a rolling Sharpe of eight monthly returns.
"""

from test_the_api_page_s_inference_section_runs import shown
from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


def test_the_statistics_section_prints_what_the_page_shows(capsys):
    blocks = sections("Quickstart", "Statistics of a run")

    assert len(blocks) == 2

    names = {}
    for block in blocks:
        exec(compile(block, str(PAGE), "exec"), names)

    lines = shown("Statistics of a run")

    assert capsys.readouterr().out.splitlines()[-len(lines):] == lines
