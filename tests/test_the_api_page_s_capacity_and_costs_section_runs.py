"""The API page's Capacity and costs section runs as written after its Quickstart, in
one namespace, as a reader copies them: a trend run at two balances, its costs swept,
and the round-trip cost where it stops making money.
"""

from test_the_api_page_s_quickstart_and_result_run import PAGE, sections


def test_the_capacity_and_costs_section_runs_after_the_quickstart(capsys):
    blocks = sections("Quickstart", "Capacity and costs")

    assert len(blocks) == 2

    names = {}
    for block in blocks:
        exec(compile(block, str(PAGE), "exec"), names)

    said  = capsys.readouterr().out.splitlines()
    table = next(k for k, line in enumerate(said) if line.startswith("balance "))

    assert [line.split()[0] for line in said[table + 1:table + 3]] == ["10,000",
                                                                       "100,000"]

    curve = next(k for k, line in enumerate(said) if line.startswith("scale "))

    assert said[curve + 3].split()[:3] == ["1", "18.7", "bps"]
    assert round(float(said[-1]), 1)   == 30.8
