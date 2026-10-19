"""The API page's Splitters section runs as written after its Quickstart, in one
namespace, as a reader copies them: its PurgedKFold trains on the bars its comment
names, and its CombinatorialPurged counts its splits.
"""

from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


def test_the_splitters_section_runs_after_the_quickstart(capsys):
    blocks = sections("Quickstart", "Splitters")

    assert len(blocks) == 2

    names = {}
    for block in blocks:
        exec(compile(block, str(PAGE), "exec"), names)

    said = capsys.readouterr().out.splitlines()
    assert said[-2:] == ["400 599 974", "28"]
