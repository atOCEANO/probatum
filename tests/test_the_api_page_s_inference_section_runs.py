"""The API page's Inference section runs as written after its Quickstart, in one
namespace, as a reader copies them: its eight monthly returns print the fields and the
frames the page shows beneath them, the Specification's worked example, and the Quickstart's run
is read at its own annualisation.
"""

from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


def shown(title):
    """The lines of the first text block in the page's section ``title``."""
    inside, keep, lines = False, False, []
    for line in PAGE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            keep = line[3:] == title
        elif keep and line.startswith("```text"):
            inside = True
        elif inside and line.startswith("```"):
            return lines
        elif inside:
            lines.append(line)

    return lines


def test_the_inference_section_prints_what_the_page_shows(capsys):
    blocks = sections("Quickstart", "Inference")

    assert len(blocks) == 2

    names = {}
    for block in blocks:
        exec(compile(block, str(PAGE), "exec"), names)

    said  = capsys.readouterr().out.splitlines()
    lines = shown("Inference")

    assert said[-len(lines):] == lines
    assert lines[0] == "1.73 from -0.92 to 4.39, lag 2, floored True"
