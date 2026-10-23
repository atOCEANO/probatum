"""The API page's Labels section runs as written after its Quickstart, in one namespace,
as a reader copies them: a label at every row of the one instrument but its last, those
whose horizon runs past the bars censored, and a splitter built on where each trade
closed.
"""

from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


def test_the_labels_section_runs_after_the_quickstart(capsys):
    blocks = sections("Quickstart", "Labels")

    assert len(blocks) == 2

    names = {}
    for block in blocks:
        exec(compile(block, str(PAGE), "exec"), names)

    labels = names["labels"]

    assert labels.ret.shape == (1_000, 1) and labels.exit[-1, 0] == ""
    assert (labels.exit[975:999, 0] == "censored").all()
    assert names["splitter"].get_n_splits() == 5

    said = capsys.readouterr().out.splitlines()

    assert "Labels(BTCUSDT over 1000 bars, long 10,000.0 at a horizon of 24)" in said
