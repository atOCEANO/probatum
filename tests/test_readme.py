"""The README's install commands must name the version that is actually shipping.

A tag hardcoded into an install line is correct on the day of a release and wrong
the day after, and nothing catches it: the old URL still resolves, to an old
wheel, so the reader installs a version behind and no error is raised anywhere.
This turns that into a failing build at the moment the version is bumped.
"""

import pathlib
import re

from packaging.version import Version

import probatum

# the gate copies it beside the docs; a source checkout has it above tests
_CANDIDATES = (
    pathlib.Path("/README.md"),
    pathlib.Path(__file__).resolve().parent.parent / "README.md",
)

# every place a release tag can hide in an install command, and the version a
# wheel or source archive repeats in its own file name; a tag spells a pre-release
# as Cargo does, 2.0.0-rc.1, and a file name as Python does, 2.0.0rc1
_PINNED = re.compile(
    r"(?:expanded_assets/v|releases/download/v|\.git@v)"
    r"(\d+\.\d+\.\d+(?:-[0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*)?)"
    r"|(?:/emsl-|/probatum-)(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?)")


def readme():
    for path in _CANDIDATES:
        if path.is_file():
            return path.read_text(encoding="utf-8")

    return None


def test_the_readme_is_where_the_tests_can_see_it():
    assert readme() is not None, "no README.md in the gate or above tests"


def test_every_pinned_version_in_the_readme_is_the_one_shipping():
    found = sorted({tag or name for tag, name in _PINNED.findall(readme())})

    assert found, "no pinned version found; the install commands may have moved"
    assert {Version(each) for each in found} == {Version(probatum.__version__)}, (
        f"README pins {found} and the package is {probatum.__version__}; "
        f"bump the install commands in the same commit as the version"
    )
