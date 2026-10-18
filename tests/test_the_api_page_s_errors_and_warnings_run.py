"""The API page's Errors and warnings section runs as written after its Quickstart, in
one namespace, as a reader copies them: each refusal it shows is caught as the class it
names, and the logging and warning settings it makes are undone after.
"""

import logging

from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


def test_the_errors_section_runs_after_the_quickstart(capsys):
    blocks = sections("Quickstart", "Errors and warnings")

    assert len(blocks) == 2

    names = {}
    root  = logging.getLogger()
    handlers, level = root.handlers[:], root.level

    try:
        for block in blocks:
            exec(compile(block, str(PAGE), "exec"), names)
    finally:
        root.handlers[:], root.level = handlers, level

    said = capsys.readouterr().out
    assert "policy: got the class Hundred, not an instance; pass Hundred()" in said
    assert "ProbatumKeyError data: no instrument 'SOLUSDT'" in said
