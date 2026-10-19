"""The API page's Validation section runs as written after its Quickstart, in one
namespace, as a reader copies them: four paths from ten fits, every window of them on
the rows it owns, and the one look at the tail counted.
"""

from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


def test_the_validation_section_runs_after_the_quickstart(capsys):
    blocks = sections("Quickstart", "Validation")

    assert len(blocks) == 2

    names = {}
    for block in blocks:
        exec(compile(block, str(PAGE), "exec"), names)

    study = names["study"]

    assert (len(study.paths), len(study.fits), study.looks) == (4, 10, 1)
    assert names["held"] is study.holdout

    said = capsys.readouterr().out

    assert "Study(4 paths of Channel from 10 fits by CombinatorialPurged(5, 2)" in said
