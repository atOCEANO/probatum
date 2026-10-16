"""The documented examples must be real code against the real signatures.

An example that names a parameter the library does not have is worse than no
example: it reads as authoritative, it is what a reader copies first, and it
fails on their first attempt rather than on ours. Nothing here runs the examples,
because most of them stand on a frame and a strategy the page describes in prose;
what it checks is that every block parses and every keyword exists.
"""

import ast
import inspect
import pathlib
import re

import probatum
from probatum import metrics, plot, ta
from probatum.backtest import Backtester

# the gate copies the pages to /docs; a source checkout has them beside tests
_CANDIDATES = (
    pathlib.Path("/docs"),
    pathlib.Path(__file__).resolve().parent.parent / ".Documentation",
)

# /dev is taken on any linux box, so the python stages land the dev tools on
# /devtools; a source checkout has them beside tests
_TOOLS = (
    pathlib.Path("/devtools"),
    pathlib.Path(__file__).resolve().parent.parent / "dev",
)

# the gate copies the Dockerfile too, so the pages can be checked against the
# stages they tell a contributor to run
_GATE = (
    pathlib.Path("/Dockerfile"),
    pathlib.Path(__file__).resolve().parent.parent / "Dockerfile",
)

# builder produces the wheel every other stage installs, and the one recipe that
# does run it alone sits in dev/golden.py's docstring rather than in a page, so
# the stage is exempt here for as long as that is where the recipe lives
_INTERNAL_STAGES = {"builder"}

_API = {
    "chart":          probatum.chart,
    "chart_defaults": probatum.chart_defaults,
    "to_ohlcv":       probatum.to_ohlcv,
    "Backtester":     Backtester,
    "Line":           plot.Line,
    "Histogram":      plot.Histogram,
    "Band":           plot.Band,
    "Level":          plot.Level,
    "Marker":         plot.Marker,
    "Markers":        plot.Markers,
    "Background":     plot.Background,
    "Panel":          plot.Panel,
    "Recorder":       plot.Recorder,
    "ramp":           plot.ramp,
    "at_bar":         plot.at_bar,
    "at_next":        plot.at_next,
    "tune":           probatum.tune,
    "walk_forward":   probatum.walk_forward,
    "Market":         probatum.Market,
    "Data":           probatum.Data,
    "from_router":    probatum.Data.from_router,
}
# every public function of the two modules whose examples are all calls. Listed by
# reflection rather than by hand: the hand-written dict above had gone four
# releases without gaining an entry, so the newest examples were the least checked
_API.update({name: getattr(metrics, name) for name in metrics.__all__
             if callable(getattr(metrics, name))})
_API.update({name: getattr(ta, name) for name in ta.__all__
             if callable(getattr(ta, name)) and not isinstance(getattr(ta, name), type)})

# a bare name, or an attribute hanging off one of these, is the library function.
# self.log.at_bar is Recorder.at_bar, which takes **values and shares its name
# with the module-level helper, so matching on the last word alone is wrong
_ROOTS = {"probatum", "plot", "metrics", "ta"}


def pages():
    for root in _CANDIDATES:
        if root.is_dir():
            found   = sorted(root.glob("*.md"))
            outside = root.parent / "README.md"
            if outside.is_file():
                found.append(outside)
            return found

    return []


def blocks(text):
    out, buf, inside = [], [], False
    for line in text.splitlines():
        if line.startswith("```python"):
            inside, buf = True, []
        elif line.startswith("```") and inside:
            out.append("\n".join(buf))
            inside = False
        elif inside:
            buf.append(line)

    return out


def called(func):
    if isinstance(func, ast.Name):
        return func.id

    if not isinstance(func, ast.Attribute):
        return None

    base = func.value
    if isinstance(base, ast.Name) and base.id in _ROOTS:
        return func.attr

    if (isinstance(base, ast.Attribute) and base.attr in _ROOTS
            and isinstance(base.value, ast.Name) and base.value.id in _ROOTS):
        return func.attr

    return None


def tools():
    for root in _TOOLS:
        if root.is_dir():
            return root
    return None


def docs_root():
    for root in _CANDIDATES:
        if root.is_dir():
            return root
    return None


def test_the_documentation_has_pages_to_check():
    assert pages(), "no .Documentation found in the gate or beside tests"


def test_every_tool_under_dev_is_named_by_a_page():
    root = tools()
    assert root is not None, "dev/ was not found in the gate or beside tests"

    text    = "\n".join(page.read_text(encoding="utf-8") for page in pages())
    missing = sorted(str(path.relative_to(root)).replace("\\", "/")
                     for path in root.rglob("*.py") if path.name not in text)

    # a directory holding no python of its own still has to be accounted for:
    # dev/diagrams is mermaid sources and two config files and no tool at all
    missing += sorted(f"{child.name}/" for child in root.iterdir()
                      if child.is_dir() and child.name != "__pycache__"
                      and child.name not in text)

    assert not missing, f"under dev/ and explained by no page: {missing}"


def decisions_page():
    for page in pages():
        if page.name == "Decisions.md":
            return page.read_text(encoding="utf-8")
    raise AssertionError("Decisions.md is not among the pages")


def declared_decisions(text):
    # a decision is a paragraph opening with its number and title in bold, and the
    # list keeps page order so that a repeat or a misplaced entry can be seen
    return [int(number) for number in re.findall(r"^\*\*(\d{4}) ", text, re.M)]


def gate_stages():
    for path in _GATE:
        if path.is_file():
            return set(re.findall(r"^FROM .* AS ([a-z0-9-]+)$",
                                  path.read_text(encoding="utf-8"), re.M))
    return None


def test_the_decisions_are_numbered_without_a_gap_or_a_repeat():
    declared = declared_decisions(decisions_page())
    assert declared, "no numbered decisions were found"

    repeated = sorted({n for n in declared if declared.count(n) > 1})
    assert not repeated, (
        f"{[f'{n:04d}' for n in repeated]} opens more than one decision; each number "
        f"belongs to one"
    )

    missing = sorted(set(range(1, max(declared) + 1)) - set(declared))
    assert not missing, (
        f"the decisions run to {max(declared):04d} but {[f'{n:04d}' for n in missing]} "
        f"is not among them; either it was lost or the numbering skipped it"
    )
    assert declared == sorted(declared), (
        "the decisions are out of number order; the page lists them in order"
    )


def test_every_decision_a_page_cites_is_one_that_exists():
    declared = set(declared_decisions(decisions_page()))
    text     = "\n".join(page.read_text(encoding="utf-8") for page in pages())
    cited    = set()

    for run in re.findall(r"ADRs?\s+((?:\d{4})(?:(?:,|\s+and)\s+\d{4})*)", text):
        cited.update(int(number) for number in re.findall(r"\d{4}", run))

    assert cited, "no decisions were cited anywhere, which cannot be right"

    dangling = sorted(cited - declared)
    assert not dangling, (
        f"pages cite {[f'{n:04d}' for n in dangling]}, which the decisions page "
        f"does not define"
    )


def test_every_link_between_the_pages_names_a_page_that_exists():
    # matched on the file name rather than the path, because the gate lays the
    # pages out flat under /docs while a checkout keeps README.md a level above
    # them, and a link is relative to whichever of those it was written in
    known  = {page.name for page in pages()}
    broken = []

    for page in pages():
        for target in re.findall(r"\]\(([^)]+\.md)(?:#[^)]*)?\)",
                                 page.read_text(encoding="utf-8")):
            if pathlib.PurePosixPath(target).name not in known:
                broken.append(f"{page.name} -> {target}")

    assert not broken, "\n".join(broken)


def test_no_link_leaves_the_repository():
    # the test above only follows .md targets, so a link to anything else was
    # unchecked, and one to a notebook in a scratch folder beside the checkout
    # shipped in 1.0.0 pointing at a path no reader could ever have. Resolving
    # these is not possible here, because the gate lays the pages out flat and a
    # checkout does not, but the depth is the same in both: the pages sit one
    # level down, so ../ reaches the repository root and ../../ is already gone
    for page in pages():
        for target in re.findall(r"\]\(([^)]+)\)", page.read_text(encoding="utf-8")):
            assert "../../" not in target, f"{page.name} -> {target}"


def test_the_documented_install_names_the_version_that_is_installed():
    # probatum is not on PyPI, so the release URL and the git tag in the README are
    # the install path rather than a convenience. A version bump is the one edit
    # always made in a hurry, and a stale one here sends a reader to a release
    # that predates everything the page around it describes. The wheel is built
    # from Cargo.toml, so this holds the README to that number without reading it
    want  = f"v{probatum.__version__}"
    wrong = []

    for page in pages():
        for found in re.findall(r"\bv\d+\.\d+\.\d+\b",
                                page.read_text(encoding="utf-8")):
            if found != want:
                wrong.append(f"{page.name} says {found}, the wheel is {want}")

    assert not wrong, "\n".join(wrong)


def test_every_gate_stage_a_contributor_can_run_is_written_down():
    stages = gate_stages()
    assert stages is not None, "no Dockerfile in the gate or beside tests"

    text         = "\n".join(page.read_text(encoding="utf-8") for page in pages())
    undocumented = sorted(
        stage for stage in stages - _INTERNAL_STAGES
        if f"--target {stage}" not in text and f"`{stage}`" not in text
    )

    assert not undocumented, f"a stage no page tells anyone about: {undocumented}"


def test_dev_keeps_no_documentation_of_its_own():
    root = tools()
    assert root is not None, "dev/ was not found in the gate or beside tests"

    strays = sorted(str(path.relative_to(root)).replace("\\", "/")
                    for path in root.rglob("*.md"))
    assert not strays, (
        f"documentation belongs in .Documentation and the top-level README, not "
        f"beside the code it describes; found {strays}"
    )


def test_every_documented_example_parses():
    broken = []

    for page in pages():
        for body in blocks(page.read_text(encoding="utf-8")):
            try:
                ast.parse(body)
            except SyntaxError as exc:
                first = (body.strip().splitlines() or [""])[0]
                broken.append(f"{page.name}: {exc.msg} near {first[:60]!r}")

    assert not broken, "\n".join(broken)


def test_every_documented_example_names_real_parameters():
    wrong = []

    for page in pages():
        for body in blocks(page.read_text(encoding="utf-8")):
            try:
                tree = ast.parse(body)
            except SyntaxError:
                continue                      # reported by the test above
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = _API.get(called(node.func))
                if target is None:
                    continue
                allowed = set(inspect.signature(target).parameters)
                for keyword in node.keywords:
                    if keyword.arg is not None and keyword.arg not in allowed:
                        wrong.append(
                            f"{page.name}: {called(node.func)}"
                            f"({keyword.arg}=) is not a parameter"
                        )

    assert not wrong, "\n".join(wrong)


def api_page():
    for page in pages():
        if page.name == "Python_API.md":
            return page.read_text(encoding="utf-8")
    raise AssertionError("Python_API.md is not among the pages")


def test_every_export_the_table_names_actually_exists():
    listed = re.findall(r"^\| `probatum.(\w+)`", api_page(), re.M)
    assert listed, "the export table was not found"

    for name in listed:
        try:
            getattr(probatum, name)
        except AttributeError as exc:
            raise AssertionError(
                f"the export table names probatum.{name}, which the package does not "
                f"have: {exc}"
            )
        except ImportError as exc:
            # a name behind an optional extra: the package has it, and says which
            # extra to install rather than denying the name exists at all
            assert "pip install" in str(exc), (
                f"probatum.{name} failed without naming its extra: {exc}"
            )


def test_nothing_is_exported_without_being_documented():
    page    = api_page()
    missing = [name for name in probatum.__all__ if f"`probatum.{name}`" not in page
               and f"`{name}`" not in page]

    assert not missing, f"exported but undocumented: {missing}"


def test_nothing_a_module_exports_is_undocumented_either():
    text    = "\n".join(page.read_text(encoding="utf-8") for page in pages())
    missing = []
    for name in ("metrics", "ta", "plot"):
        for export in getattr(probatum, name).__all__:
            # named in prose as `export`, called in an example as export(, or
            # reached through its module either way
            if not any(mark in text for mark in
                       (f"`{export}", f"{name}.{export}", f"{export}(")):
                missing.append(f"{name}.{export}")

    assert not missing, f"exported by a module and documented nowhere: {missing}"


def test_the_indicator_count_the_docs_claim_is_the_count_there_is():
    functions = [n for n in ta.__all__ if not isinstance(getattr(ta, n), type)]
    page      = api_page()
    listed    = set(re.findall(r"`(\w+)\(", page.split("## Indicators")[1]))
    uncovered = sorted(set(functions) - listed)

    assert not uncovered, f"indicators missing from the API table: {uncovered}"

    for name in ("Python_API.md", "README.md"):
        text = [p for p in pages() if p.name == name][0].read_text(encoding="utf-8")
        assert f"{len(functions)} functions" in text or f"{len(functions)} indicators" in text, (
            f"{name} does not state the indicator count of {len(functions)}"
        )


def test_every_chart_image_that_ships_was_drawn_by_a_call_in_the_builder():
    root = docs_root()
    assert root is not None, "no .Documentation found in the gate or beside tests"

    dev = tools()
    assert dev is not None, "dev/ was not found in the gate or beside tests"

    shipped = sorted((root / "imgs" / "charts").glob("*.png"))
    build   = (dev / "charts" / "build.py").read_text(encoding="utf-8")
    drawn   = re.findall(r'\bkeep\("', build)

    assert len(shipped) == len(drawn), (
        f"{len(shipped)} chart images ship and build.py draws {len(drawn)}"
    )
