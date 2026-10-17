"""The API page's Quickstart and its Result section run as written, one after another in
one namespace, as a reader copies them: the Quickstart builds the Data and the Result
the section reads and queries. Every block of the page is checked for
its names and keywords by ``test_docs.py``; these are run.
"""

import pathlib

import pytest

# the gate stages copy the doc set to /docs, and a source checkout, which is what
# CI runs from, has it beside tests
_CANDIDATES = (
    pathlib.Path("/docs/Python_API.md"),
    pathlib.Path(__file__).resolve().parent.parent / ".Documentation" / "Python_API.md",
)
PAGE = next((path for path in _CANDIDATES if path.is_file()), None)

if PAGE is None:
    pytest.skip("the documentation is not here", allow_module_level=True)


def sections(*titles):
    """The python blocks of the page's sections ``titles``, a section running from its
    heading to the next one of its level, in the order the page gives them."""
    found, inside, block, keep = [], False, [], False
    for line in PAGE.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            keep = line[3:] in titles
        elif line.startswith("```python"):
            inside, block = True, []
        elif line.startswith("```") and inside:
            inside = False
            if keep:
                found.append("\n".join(block))
        elif inside:
            block.append(line)

    return found


def test_the_quickstart_and_the_result_section_run_as_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    blocks = sections("Quickstart", "The Result")

    assert len(blocks) >= 3

    names = {}
    for block in blocks:
        exec(compile(block, str(PAGE), "exec"), names)

    result = names["result"]

    assert len(result.fills) > 0
