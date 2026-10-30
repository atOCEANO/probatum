"""The API page's Validation section runs as written after its Quickstart, in one
namespace, as a reader copies them: four paths from ten fits, every window of them on
the rows it owns, and the one look at the tail counted; then a model's validation,
seven paths from 28 fits with its forecast.
"""

import pytest

from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


def test_the_validation_section_runs_after_the_quickstart(capsys):
    pytest.importorskip("sklearn")

    blocks = sections("Quickstart", "Validation")

    assert len(blocks) == 3

    names = {}
    for block in blocks[:2]:
        exec(compile(block, str(PAGE), "exec"), names)

    study = names["study"]

    assert (len(study.paths), len(study.fits), study.looks) == (4, 10, 1)
    assert names["held"] is study.holdout

    said = capsys.readouterr().out

    assert "Study(4 paths of Channel from 10 fits by CombinatorialPurged(5, 2)" in said

    exec(compile(blocks[2], str(PAGE), "exec"), names)
    study = names["study"]

    assert (len(study.paths), len(study.fits), study.looks) == (7, 28, 1)
    assert names["held"] is study.holdout and study.forecast is not None
    assert "\nrank_ic " in capsys.readouterr().out
